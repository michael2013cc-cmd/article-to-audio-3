#!/usr/bin/env python3
"""Companion transcript with timestamps.

A long spoken lecture is hard to search by ear. Emitting the exact text
that was spoken, with a start time for every sentence, gives the listener
an index they can grep - which paragraphs came from the source document
and which were added by the narrator, in particular.

v2 could only time the chapters. v3 times every sentence, because
edge-tts reports where each word is spoken and the script can be walked
in parallel with those words (see ``align``).
"""

from __future__ import annotations


def clock(seconds: float) -> str:
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def build_transcript(
    chapters,
    anchors: list[dict],
    sentences_by_chapter: list[list[dict]] | None = None,
    *,
    audio_name: str,
    title: str = "",
    header_notes: list[str] | None = None,
    separator_width: int = 62,
) -> str:
    """Build the timestamped transcript.

    ``anchors`` is a list of ``{"start": float, "sec": float, "title": str}``.
    ``sentences_by_chapter`` holds one list of
    ``{"text", "start", "end"}`` per chapter, with absolute times.
    """
    lines: list[str] = []
    total = anchors[-1]["start"] + anchors[-1]["sec"] if anchors else 0.0

    lines.append(title.strip())
    lines.append("")
    lines.append(f"对应音频：{audio_name}")
    lines.append(f"总时长：{clock(total)}（约 {total / 60:.0f} 分钟）")
    lines.append(f"章节标记：共 {len(anchors)} 节，已写入 MP3 的 ID3v2.4 CHAP/CTOC 标签。")
    for note in header_notes or []:
        lines.append(note)

    lines += ["", "【时间点索引】"]
    for i, ch in enumerate(anchors):
        lines.append(f"  {clock(ch['start'])}  第{i + 1}节　{ch['title']}")

    lines += ["", "=" * separator_width]
    for i, ch in enumerate(anchors):
        lines += ["", f"【{clock(ch['start'])}】第{i + 1}节　{ch['title']}"
                      f"　（本节约 {clock(ch['sec'])}）", ""]
        body = chapters[i].body
        sentences = (sentences_by_chapter[i] if sentences_by_chapter and i < len(sentences_by_chapter) else [])
        if sentences:
            for sent in sentences:
                lines.append(f"  {clock(sent['start'])}  {sent['text']}")
                lines.append("")
        else:
            for para in body.split("\n\n"):
                para = para.strip()
                if para:
                    lines += [para, ""]
        lines.append("-" * separator_width)

    return "\n".join(lines)


def build_word_transcript(
    chapters,
    anchors: list[dict],
    words_by_chapter: list[list[dict]] | None = None,
    *,
    audio_name: str,
    title: str = "",
    separator_width: int = 62,
) -> str:
    """A word-by-word script, for checking a single figure by ear.

    Each line holds one word and the second it is spoken at.
    """
    lines: list[str] = []
    total = anchors[-1]["start"] + anchors[-1]["sec"] if anchors else 0.0
    lines.append(title.strip())
    lines.append("")
    lines.append(f"对应音频：{audio_name}")
    lines.append(f"总时长：{clock(total)}（约 {total / 60:.0f} 分钟）")
    lines += ["", "=" * separator_width]
    for i, ch in enumerate(anchors):
        lines += ["", f"【{clock(ch['start'])}】第{i + 1}节　{ch['title']}", ""]
        words = (words_by_chapter[i] if words_by_chapter and i < len(words_by_chapter) else [])
        row = []
        for w in words:
            row.append(f"{clock(w['start'])} {w['text']}")
        # ten words per line keeps the column narrow enough to read
        for j in range(0, len(row), 10):
            lines.append("  " + "　".join(row[j:j + 10]))
        lines.append("-" * separator_width)
    return "\n".join(lines)


def write_transcript(path: str, text: str) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path
