#!/usr/bin/env python3
"""Text-to-speech through Microsoft Edge's neural voices (via edge-tts).

One request per chapter. The endpoint is a free public service with no
guarantees, so every call is wrapped in a retry loop with backoff.

WHAT IS NEW HERE COMPARED TO v2
v2 synthesised each chapter and threw the timing information away, so the
companion transcript could only say "this chapter starts at 04:12".

edge-tts can be asked for WORD boundaries instead of sentence boundaries -
the default is sentence boundaries, but passing ``boundary="WordBoundary"``
makes the service report the start and duration of every word it speaks.
Measured on this machine: a two-sentence paragraph came back with 28 word
marks, e.g. ``12276`` at 5.652 s for 0.851 s.

Those marks are what make the timestamped transcript precise, and they are
free: they come out of the same synthesis call, not a second pass.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict, dataclass

import edge_tts
from edge_tts.constants import TICKS_PER_SECOND

DEFAULT_VOICE = "zh-CN-YunyangNeural"  # news-anchor style, steady for long-form
DEFAULT_RATE = "-6%"  # a slow, comfortable pace for spoken numbers and lists

MAX_ATTEMPTS = 5
BACKOFF = 5.0

# "WordBoundary" gives word-level marks; "SentenceBoundary" is the service
# default. Both are accepted on the way back, because a service-side change
# should degrade the output, not break the run.
BOUNDARY = "WordBoundary"


@dataclass
class WordMark:
    """One spoken word and where it sits in the chapter's audio."""

    start: float      # seconds, relative to the start of the chapter
    duration: float   # seconds
    text: str

    @property
    def end(self) -> float:
        return self.start + self.duration


async def _stream(text: str, path: str, voice: str, rate: str, boundary: str):
    """Synthesise one block, collecting audio and boundary marks."""
    communicate = edge_tts.Communicate(text, voice, rate=rate, boundary=boundary)
    marks: list[WordMark] = []
    with open(path, "wb") as fh:
        async for chunk in communicate.stream():
            kind = chunk.get("type")
            if kind == "audio":
                fh.write(chunk["data"])
            elif kind in ("WordBoundary", "SentenceBoundary"):
                offset = chunk.get("offset")
                duration = chunk.get("duration")
                if offset is None:
                    continue
                marks.append(WordMark(
                    start=offset / TICKS_PER_SECOND,
                    duration=(duration or 0) / TICKS_PER_SECOND,
                    text=chunk.get("text") or "",
                ))
    return marks


def synth_text(
    text: str,
    path: str,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    attempts: int = MAX_ATTEMPTS,
    boundary: str = BOUNDARY,
) -> list[WordMark]:
    """Synthesise one block of text to an mp3 file.

    Returns the word marks for the block. Retries on network faults.
    """
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            marks = asyncio.run(_stream(text, path, voice, rate, boundary))
            if not os.path.getsize(path):
                raise RuntimeError("empty audio returned")
            return marks
        except Exception as exc:  # noqa: BLE001 - the endpoint fails in many ways
            last = exc
            print(f"  synth attempt {attempt}/{attempts} failed: {exc!r}", flush=True)
            if attempt < attempts:
                time.sleep(BACKOFF)
    raise RuntimeError(f"TTS failed after {attempts} attempts: {last}")


def _marks_path(workdir: str, index: int) -> str:
    return os.path.join(workdir, f"marks_{index}.json")


def save_marks(path: str, marks: list[WordMark]) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([asdict(m) for m in marks], fh, ensure_ascii=False)
    return path


def load_marks(path: str) -> list[WordMark]:
    with open(path, encoding="utf-8") as fh:
        return [WordMark(**row) for row in json.load(fh)]


def synth_chapters(
    chapters,
    workdir: str,
    voice: str = DEFAULT_VOICE,
    rate: str = DEFAULT_RATE,
    skip_existing: bool = False,
    boundary: str = BOUNDARY,
) -> list[dict]:
    """Synthesise each chapter to ``workdir/raw_<n>.mp3``.

    Returns one record per chapter: ``{"path": str, "marks": [WordMark]}``.

    ``skip_existing`` makes the run resumable after a dropped connection.
    The word marks are cached next to the audio, so a resumed run keeps the
    timestamp precision of the original synthesis.
    """
    os.makedirs(workdir, exist_ok=True)
    out = []
    for i, ch in enumerate(chapters):
        path = os.path.join(workdir, f"raw_{i}.mp3")
        marks_file = _marks_path(workdir, i)
        if skip_existing and os.path.exists(path) and os.path.getsize(path):
            print(f"[{i}] {ch.title}: kept existing audio", flush=True)
            marks = load_marks(marks_file) if os.path.exists(marks_file) else []
        else:
            t0 = time.time()
            marks = synth_text(ch.body, path, voice, rate, boundary=boundary)
            save_marks(marks_file, marks)
            print(
                f"[{i}] {ch.title}: {os.path.getsize(path) // 1024} KiB, "
                f"{len(marks)} word marks, in {time.time() - t0:.0f}s",
                flush=True,
            )
        out.append({"path": path, "marks": marks})
    return out
