#!/usr/bin/env python3
"""Listen back and check the figures that were actually spoken.

Combines the local speech-to-text pass (:mod:`a2a.stt`) with the key
matching ported from v1 (:mod:`a2a.match`).

RECHECK - a miss is not a verdict (the lesson v1 paid for three times)
Whisper, like v1's platform STT, can mishear a figure that was spoken
perfectly, so a single transcript cannot prove the audio wrong. When a key
misses, the recheck transcribes ONLY the window where the script says the
figure sits - the pipeline knows every sentence's timestamps - with a
different decoder configuration (beam_size=1, cut at the sentence instead
of wherever VAD happened to cut):

    window re-read contains the key -> pass 1 was recognition noise; the
                                      audio has the figure. Not an error.
    window re-read misses, and BOTH
    passes agree on the SAME other
    number                          -> the audio itself is damaged;
                                      re-render that chapter.
    anything else                   -> inconclusive: re-run listen, or
                                      check that stretch by ear.

Without sentence times (the standalone ``listen`` command) there is no
window to zoom into, so the second pass covers the whole file and the
"agrees on the same wrong value" test is skipped: on a full transcript,
some OTHER paragraph almost always shares a number, and agreement there
proves nothing.
"""

from __future__ import annotations

import json
import os

from . import keys as keymod
from . import match, stt
from .numbers import normalize_script
from .transcript import clock


def check_content(
    path: str,
    keys: list,
    model_size: str = stt.DEFAULT_MODEL,
    device: str = "auto",
    transcript_path: str | None = None,
    quiet: bool = False,
    sentences: list | None = None,
    recheck: bool = True,
    transcribe=None,
) -> dict:
    """Transcribe the finished audio and check every key against it.

    ``sentences`` is the pipeline's per-sentence timeline
    (``{"text", "start", "end", "chapter", "chapter_title"}``); it is what
    lets the recheck zoom into the exact seconds a missing figure should
    occupy. ``transcribe`` is injectable so the recheck logic can be tested
    without loading a model.

    Returns a report with the transcript, the per-key verdict and the list
    of missing keys. ``ok`` is True only when every key was heard.
    """
    if not keys:
        return {
            "ok": False,
            "skipped": True,
            "reason": "no keys to check",
            "total": 0,
            "found": 0,
            "missing": [],
        }

    if transcribe is None:
        # Load the model ONCE: the recheck would otherwise pay the
        # multi-GB load a second time for the same file.
        model_obj, dev, compute = stt.load_model(model_size, device)
        if not quiet:
            print(f"  model {model_size} on {dev}/{compute}", flush=True)

        def transcribe(p, *, clip_timestamps=None, beam_size=5):
            return stt.transcribe(
                p, model=model_obj, progress=False,
                clip_timestamps=clip_timestamps, beam_size=beam_size,
            )

    result = transcribe(path)
    text = result["text"]
    if transcript_path:
        with open(transcript_path, "w", encoding="utf-8") as fh:
            fh.write(text)

    report = match.match_report(text, keys)
    report["model"] = model_size

    if report["missing"] and recheck:
        _recheck(report, result, path, sentences, transcribe, quiet)

    report["transcript_path"] = transcript_path
    report["segments"] = result["segments"]
    report["transcript"] = text
    if not quiet:
        print(
            f"      回听：{report['found']}/{report['total']} 个关键数字命中",
            flush=True,
        )
        for row in report["missing"]:
            print(f"      !! 没听到：{row['key']}", flush=True)
    return report


def _local_text(first: dict, window: dict) -> str:
    """What the FIRST pass heard inside the window, from its segments."""
    start = window["start"]
    end = window.get("end")
    parts = []
    for seg in first.get("segments", []):
        s, e = seg["start"], seg["end"]
        if s < (end if end is not None else float("inf")) and e > start:
            parts.append(seg["text"])
    return "".join(parts)


def _figures(text: str) -> set[str]:
    """Every numeric candidate label in a transcript fragment."""
    if not text:
        return set()
    return {
        c["key"].replace(",", "")
        for c in keymod.extract_candidates([text])
    }


def _windows_for(forms: list[str], sentences: list | None) -> list[dict]:
    """The sentences a key's accepted forms literally appear in.

    The keys were derived FROM the script, so some rendering of the figure
    is in the sentence text itself - Arabic ("12276") or the Chinese run it
    was converted from. A key whose forms match no sentence falls back to
    the whole file.
    """
    if not sentences:
        return []
    out = []
    for s in sentences:
        text = s["text"]
        norm = normalize_script(text)
        if any(f and (f in text or f in norm) for f in forms):
            out.append(s)
    out.sort(key=lambda s: s["start"])
    return out


def _recheck(report, first, path, sentences, transcribe, quiet):
    """Second opinion on every missing key; mutates the report rows."""
    def say(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    if not sentences:
        say("      复验：无稿内时间点，整档重听一遍（beam=1）…")
        second_full = transcribe(path, beam_size=1)["text"]

    recovered: list[str] = []
    damaged: list[str] = []
    inconclusive: list[str] = []

    for row in report["missing"]:
        label = row["key"]
        forms = [f.replace(",", "") for f in row["alts"]]

        if not sentences:
            # One shared second pass; no window means no way to localise an
            # "agreed wrong value", so everything un-recovered is honestly
            # inconclusive rather than guessed at.
            if match.match_keys(second_full, [row["source"]]) == []:
                row.update(via="recheck", verdict="recovered_on_recheck")
                row["found"] = True
                recovered.append(label)
                say(f"      复验命中：{label}（第一次是识别噪声）")
            else:
                row.update(via="recheck", verdict="inconclusive")
                inconclusive.append(label)
            continue

        windows = _windows_for(forms, sentences)
        if not windows:
            windows = [{"start": 0.0, "end": None, "text": "",
                        "chapter": None, "chapter_title": ""}]
        say(f"      复验 {label} …")
        hit = None
        last = None
        for w in windows:
            # faster-whisper takes clip_timestamps as "start,end" seconds
            # w.r.t. the input file; our sentence times are already that.
            clip = (
                f"{w['start']},{w['end']}"
                if w.get("end") is not None else None
            )
            local2 = transcribe(path, clip_timestamps=clip, beam_size=1)["text"]
            local1 = _local_text(first, w)
            if match.match_keys(local2, [row["source"]]) == []:
                hit = local2
                break
            last = (local1, local2, w)
        if hit is not None:
            row.update(found=True, via="recheck", verdict="recovered_on_recheck")
            recovered.append(label)
            say(f"      复验命中：{label} 在第二次读法里听到了（第一次是识别噪声）")
            continue

        local1, local2, w = last
        agreed = sorted((_figures(local1) & _figures(local2)) - set(forms))
        where = w.get("chapter_title") or "未知章节"
        span = (w["start"], w["end"] if w.get("end") is not None else first.get("duration", 0.0))
        row.update(
            via="recheck",
            verdict="suspect_audio" if agreed else "inconclusive",
            evidence="、".join(agreed[:3]) or None,
            chapter=w.get("chapter"),
            chapter_title=w.get("chapter_title"),
            window=(round(span[0], 1), round(span[1], 1)),
        )
        if agreed:
            damaged.append(label)
            say(f"      !! {label} 两遍都读成 {agreed[0]} —— 音频疑似损坏（{where}）")
        else:
            inconclusive.append(label)
            say(f"      ?? {label} 两遍都没听到且无一致替代读数 —— "
                f"请人工复听 {clock(span[0])}–{clock(span[1])}")

    report["found"] = sum(1 for r in report["rows"] if r["found"])
    report["missing"] = [r for r in report["rows"] if not r["found"]]
    report["ok"] = all(r["found"] for r in report["rows"])
    report["recheck"] = {
        "recovered": recovered,
        "suspect_audio": damaged,
        "inconclusive": inconclusive,
    }


def load_keys(path: str) -> list:
    """Read a key file.

    Accepts a JSON list of strings or of ``{"key": ..., "alts": [...]}``
    objects, and also a JSON object of that shape under "keys".
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("keys", [])
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of keys")
    return data


def save_keys(path: str, keys: list) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(keys, fh, ensure_ascii=False, indent=2)
    return path
