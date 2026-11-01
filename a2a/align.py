#!/usr/bin/env python3
"""Align the spoken words back to the script, so every sentence gets a time.

edge-tts reports where each word is spoken, but not which sentence the word
belongs to. The words themselves are contiguous pieces of the script with
the punctuation removed ("先" / "看硬" / "指标"), so the script can be
walked in parallel with the word list.

That gives two things v2 could not produce:

  * a sentence-level timestamp for the companion transcript, and
  * a way to point at the exact second where a figure was spoken, which is
    what makes a missed key diagnosable instead of just red.

The alignment is greedy: for each word, look for it at the current
position in the script, and if it is not there, look ahead. Words that
cannot be placed are kept with a position of -1 so the caller can tell
"aligned" from "not aligned".
"""

from __future__ import annotations

# Characters that end a spoken sentence.
SENTENCE_END = "。！？!?；;\n"
# Characters that carry no spoken content and are skipped while matching.
SKIPPED = set(" \t　，。、；：？！「」『』（）()《》〈〉【】…—·,.!?;:\"'“”‘’\n\r")


def align(body: str, marks: list) -> list[dict]:
    """Attach the script position of every word mark.

    Returns a list of dicts with the mark's fields plus ``pos`` (the
    character offset in ``body``, or -1 when the word could not be placed).
    """
    out: list[dict] = []
    pos = 0
    n = len(body)
    for mark in marks:
        word = (mark["text"] if isinstance(mark, dict) else mark.text) or ""
        if not word:
            continue
        found = -1
        # Prefer an exact hit at the current position.
        if body.startswith(word, pos):
            found = pos
        else:
            # Look ahead, skipping punctuation and whitespace.
            scan = pos
            limit = min(n, pos + 200)
            while scan < limit:
                if body.startswith(word, scan):
                    found = scan
                    break
                if body[scan] not in SKIPPED:
                    break  # a real character sits in the way: do not jump over it
                scan += 1
        if found >= 0:
            pos = found + len(word)
        out.append({
            "start": mark["start"] if isinstance(mark, dict) else mark.start,
            "duration": mark["duration"] if isinstance(mark, dict) else mark.duration,
            "text": word,
            "pos": found,
        })
    return out


def sentence_spans(body: str) -> list[tuple[int, int]]:
    """Split the script body into sentence spans ``(start, end)``."""
    spans: list[tuple[int, int]] = []
    start = 0
    for i, ch in enumerate(body):
        if ch in SENTENCE_END:
            spans.append((start, i + 1))
            start = i + 1
    if start < len(body):
        spans.append((start, len(body)))
    return [(s, e) for s, e in spans if body[s:e].strip()]


def sentences(body: str, marks: list) -> list[dict]:
    """Every sentence of the body with its time span in the audio.

    A sentence that carries no word mark (an empty line, a stray symbol)
    is skipped: there is nothing to point at.
    """
    aligned = align(body, marks)
    placed = [m for m in aligned if m["pos"] >= 0]
    out: list[dict] = []
    for s, e in sentence_spans(body):
        inside = [m for m in placed if s <= m["pos"] < e]
        if not inside:
            continue
        text = body[s:e].strip()
        if not text:
            continue
        out.append({
            "text": text,
            "start": min(m["start"] for m in inside),
            "end": max(m["start"] + m["duration"] for m in inside),
            "chars": len(text),
        })
    return out


def word_marks(body: str, marks: list) -> list[dict]:
    """The aligned word list, ready to be written as a word-level script."""
    return align(body, marks)
