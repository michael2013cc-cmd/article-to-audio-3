#!/usr/bin/env python3
"""End-to-end pipeline: listening script -> tagged mp3 + transcript + check.

Stages, each of which writes into ``workdir`` so a crashed run can resume:

    script.txt
      -> raw_<i>.mp3      synthesize one chapter per request, keeping the
                         word-level timings edge-tts reports
      -> norm_<i>.mp3     loudness + fixed codec parameters
      -> <output>.mp3     lossless concatenation
      -> (tagged)         ID3v2.4 CHAP/CTOC chapter table
      -> <output>.txt     transcript with a timestamp per sentence
      -> (optional)       listen back and check the verification keys

Anchors are measured on the normalized parts, never on the raw ones: the
limiter and the encoder shift durations by a few tens of milliseconds per
chapter, and that drift accumulates into visibly wrong jump points.
"""

from __future__ import annotations

import os
import shutil

from . import align, audio, content, script as scriptmod, stt, synth, tags
from . import transcript as transcriptmod
from . import verify as verifymod
from .lock import WorkdirLock


def _chapter_times(chapters, raw_paths, norm_paths, marks_by_chapter):
    """Global sentence times for every chapter.

    The word marks come from the raw audio; the chapters are laid out on
    the normalized audio, which is a few tens of milliseconds longer per
    chapter. The ratio rescales the in-chapter times so they land on the
    right second in the finished file.
    """
    out = []
    pos = 0.0
    for i, ch in enumerate(chapters):
        norm_sec = audio.duration(norm_paths[i])
        raw_sec = audio.duration(raw_paths[i]) or norm_sec
        scale = (norm_sec / raw_sec) if raw_sec else 1.0
        sents = align.sentences(ch.body, marks_by_chapter[i])
        out.append({
            "start": pos,
            "sec": norm_sec,
            "title": ch.title,
            "sentences": [
                {
                    "text": s["text"],
                    "start": pos + s["start"] * scale,
                    "end": pos + s["end"] * scale,
                }
                for s in sents
            ],
            "words": [
                {
                    "text": w["text"],
                    "start": pos + w["start"] * scale,
                    "duration": w["duration"] * scale,
                }
                for w in align.word_marks(ch.body, marks_by_chapter[i])
            ],
        })
        pos += norm_sec
    return out


def run(
    script_path: str,
    output: str,
    *,
    workdir: str | None = None,
    voice: str = synth.DEFAULT_VOICE,
    rate: str = synth.DEFAULT_RATE,
    bitrate: str = audio.BITRATE,
    title: str | None = None,
    album: str | None = None,
    artist: str | None = None,
    year: str | None = None,
    resume: bool = False,
    write_transcript: bool = True,
    word_level: bool = False,
    keys: list | None = None,
    keys_path: str | None = None,
    stt_model: str = stt.DEFAULT_MODEL,
    stt_device: str = "auto",
    quiet: bool = False,
) -> dict:
    """Build the audio file. Returns the verify report plus the paths written.

    The workdir is located by OUTPUT NAME, not by run: two processes
    building the same output share raw_/norm_/joined files and overwrite
    each other's results. The lock refuses the second one (WorkdirBusy ->
    exit code 5) instead of racing.
    """
    wd = workdir or (os.path.splitext(os.path.abspath(output))[0] + "_work")
    lock = WorkdirLock(wd)
    lock.acquire()
    try:
        return _run(
            script_path, output,
            workdir=wd, voice=voice, rate=rate, bitrate=bitrate,
            title=title, album=album, artist=artist, year=year,
            resume=resume, write_transcript=write_transcript,
            word_level=word_level, keys=keys, keys_path=keys_path,
            stt_model=stt_model, stt_device=stt_device, quiet=quiet,
        )
    finally:
        lock.release()


def _run(
    script_path: str,
    output: str,
    *,
    workdir: str,
    voice: str = synth.DEFAULT_VOICE,
    rate: str = synth.DEFAULT_RATE,
    bitrate: str = audio.BITRATE,
    title: str | None = None,
    album: str | None = None,
    artist: str | None = None,
    year: str | None = None,
    resume: bool = False,
    write_transcript: bool = True,
    word_level: bool = False,
    keys: list | None = None,
    keys_path: str | None = None,
    stt_model: str = stt.DEFAULT_MODEL,
    stt_device: str = "auto",
    quiet: bool = False,
) -> dict:
    """Build the audio file. Returns the verify report plus the paths written."""

    def say(*args) -> None:
        if not quiet:
            print(*args, flush=True)

    chapters = scriptmod.load(script_path)
    os.makedirs(workdir, exist_ok=True)
    say(f"{len(chapters)} chapters -> {output}  (workdir {workdir})")

    say("\n[1/6] synthesize")
    synth_result = synth.synth_chapters(chapters, workdir, voice, rate, skip_existing=resume)
    raw = [r["path"] for r in synth_result]
    marks = [r["marks"] for r in synth_result]

    say("\n[2/6] normalise loudness")
    norm = audio.normalize_chapters(raw, workdir, bitrate=bitrate)

    say("\n[3/6] concat")
    joined = os.path.join(workdir, "joined.mp3")
    audio.concat(norm, joined, workdir)

    timelines = _chapter_times(chapters, raw, norm, marks)
    anchors = [{"start": t["start"], "sec": t["sec"], "title": t["title"]} for t in timelines]
    total = sum(a["sec"] for a in anchors)
    say(f"      {total / 60:.1f} min of audio, {os.path.getsize(joined) // 1024 // 1024} MiB")

    # copy the audio into place *before* tagging: mutagen only patches an
    # existing file, and saving a tag onto a missing path produces a
    # tag-only stub that looks like a success.
    shutil.copyfile(joined, output)

    say("\n[4/6] write chapter tags")
    tags.write_chapters(
        output,
        anchors,
        title=title or os.path.splitext(os.path.basename(output))[0],
        album=album,
        artist=artist,
        year=year,
    )

    say("\n[5/6] verify")
    report = verifymod.check_file(output, expect_chapters=len(chapters))
    say(
        f"      {report['integrated_lufs']} LUFS, true peak {report['true_peak_dbfs']} dBFS, "
        f"flat factor {report['flat_factor']}, chapters {report['chapters']}"
    )
    if not report["ok"]:
        for problem in report["problems"]:
            say(f"      !! {problem}")
    else:
        say("      ok")

    transcript_path = None
    if write_transcript:
        transcript_path = os.path.splitext(output)[0] + ".txt"
        header_notes = []
        text = transcriptmod.build_transcript(
            chapters,
            anchors,
            [t["sentences"] for t in timelines],
            audio_name=os.path.basename(output),
            title=title or "",
            header_notes=header_notes,
        )
        transcriptmod.write_transcript(transcript_path, text)
        say(f"      transcript -> {transcript_path}")

        if word_level:
            word_path = os.path.splitext(output)[0] + ".words.txt"
            word_text = transcriptmod.build_word_transcript(
                chapters,
                anchors,
                [t["words"] for t in timelines],
                audio_name=os.path.basename(output),
                title=title or "",
            )
            transcriptmod.write_transcript(word_path, word_text)
            say(f"      word script -> {word_path}")
            report["word_script"] = word_path

    content_report = None
    if keys:
        say("\n[6/6] listen back and check the keys")
        stt_path = os.path.splitext(output)[0] + ".heard.txt"
        # The sentence timeline is what lets a missed key be re-checked in
        # its own seconds instead of re-transcribing the whole file - and
        # what names the chapter to re-render if the audio really is wrong.
        sentence_timeline = [
            {
                "text": s["text"],
                "start": s["start"],
                "end": s["end"],
                "chapter": i,
                "chapter_title": t["title"],
            }
            for i, t in enumerate(timelines)
            for s in t["sentences"]
        ]
        content_report = content.check_content(
            output,
            keys,
            model_size=stt_model,
            device=stt_device,
            transcript_path=stt_path,
            quiet=quiet,
            sentences=sentence_timeline,
        )
        say(f"      heard transcript -> {stt_path}")
        for row in content_report.get("missing", []):
            if row.get("verdict") == "suspect_audio" and row.get("chapter") is not None:
                # edge-tts is free: re-rendering one chapter costs one
                # network call. Deleting its raw file + --resume does
                # exactly that and nothing else.
                raw = os.path.join(workdir, f"raw_{row['chapter']}.mp3")
                say(f"      修复：删除 {raw} 后加 --resume 重跑，只重做第 "
                    f"{row['chapter'] + 1} 章《{row.get('chapter_title') or ''}》")
        report["content"] = content_report

    report.update({
        "output": output,
        "transcript": transcript_path,
        "anchors": anchors,
        "duration_sec": total,
        "workdir": workdir,
    })
    if keys_path:
        report["keys_path"] = keys_path
    return report
