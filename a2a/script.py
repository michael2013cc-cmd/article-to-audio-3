#!/usr/bin/env python3
"""Listening-script format: plain text with chapter markers.

A script is a sequence of chapters. Each chapter starts with a marker line:

    @@P0|章节标题

Everything until the next marker (or end of file) is that chapter's body.
Blank lines separate paragraphs. The body is read aloud verbatim, so write
the way you want it spoken, not the way you want it typeset.

The id part (``P0``) is only a label; titles are what show up in the player's
chapter list.
"""

from __future__ import annotations

import re

MARKER = re.compile(r"^@@(?P<id>[^|\s]*)\|(?P<title>.*)$")


class Chapter:
    def __init__(self, cid: str, title: str, body: str):
        self.id = cid
        self.title = title.strip()
        self.body = body.strip()

    def __repr__(self) -> str:  # pragma: no cover
        return f"Chapter({self.id!r}, {self.title!r}, {len(self.body)} chars)"


def parse(text: str) -> list[Chapter]:
    """Parse a listening script into chapters.

    Raises ValueError when a marker line is malformed or when no marker is
    found, because a silent mis-parse would produce a single garbage chapter.
    """
    chapters: list[Chapter] = []
    cid = title = None
    buf: list[str] = []

    for lineno, line in enumerate(text.splitlines(), 1):
        if line.startswith("@@"):
            if cid is not None:
                chapters.append(Chapter(cid, title, "\n".join(buf)))
            m = MARKER.match(line.rstrip())
            if not m:
                raise ValueError(
                    f"line {lineno}: bad marker {line!r}; "
                    "expected the form  @@ID|Title"
                )
            cid, title = m.group("id"), m.group("title")
            buf = []
        else:
            if cid is None:
                if line.strip():
                    raise ValueError(
                        f"line {lineno}: body text before the first @@ID|Title marker"
                    )
                continue
            buf.append(line)

    if cid is None:
        raise ValueError("no chapter markers found; a script needs at least one")
    chapters.append(Chapter(cid, title, "\n".join(buf)))
    return chapters


def dump(chapters: list[Chapter]) -> str:
    """Serialize back to script form (round-trips with parse)."""
    return "\n\n".join(f"@@{c.id}|{c.title}\n{c.body}" for c in chapters) + "\n"


def load(path: str) -> list[Chapter]:
    with open(path, encoding="utf-8") as fh:
        return parse(fh.read())
