#!/usr/bin/env python3
"""Write ID3v2.4 chapter frames (CHAP/CTOC) and basic tags into the mp3.

Chapters are what make an hour-long lecture navigable: the player shows a
list and jumps per section instead of forcing the listener to scrub.

Two API traps worth remembering, both of which cost real time:
  * text frames take ``text=``, not ``desc=`` (``desc=`` silently yields an
    empty frame list);
  * save with ``v2_version=4`` - there is no ``v2_4=`` keyword.

A third, harmless one: mutagen upgrades the v2.3 ``TYER`` frame to ``TDRC``
when saving v2.4, so read the year back as ``TDRC``.
"""

from __future__ import annotations

from mutagen.id3 import CHAP, CTOC, TALB, TCON, TIT2, TPE1, TRCK, TYER, ID3

# CTOCFlags.TOP_LEVEL - the bit that marks the root of the chapter table.
#
# mutagen's CTOC frame has no `is_top` and no `child` keyword: the children
# live in `child_element_ids` and the top-level bit lives in the `flags`
# byte. An earlier version of this file passed `is_top=True` and `child=ids`,
# which mutagen accepted as stray keyword arguments and silently discarded -
# so every CTOC was written with an EMPTY child list and flags=0. The chapter
# table of contents listed nothing at all, and only players that scan CHAP
# frames directly ever showed the chapters.
CTOC_TOP_LEVEL = 0x02


def _text_frame(cls, value: str):
    return cls(encoding=3, text=value)


def write_chapters(
    path: str,
    anchors: list[dict],
    title: str | None = None,
    album: str | None = None,
    artist: str | None = None,
    year: str | None = None,
) -> None:
    """Tag ``path`` with a chapter table of contents.

    ``anchors`` is a list of ``{"start": float, "sec": float, "title": str}``
    in seconds, measured on the final concatenated file.
    """
    tags = ID3()
    tags.clear()

    if title:
        tags.add(_text_frame(TIT2, title))
    if album:
        tags.add(_text_frame(TALB, album))
    if artist:
        tags.add(_text_frame(TPE1, artist))
    tags.add(_text_frame(TCON, "Education"))
    tags.add(_text_frame(TRCK, "1/1"))
    if year:
        tags.add(_text_frame(TYER, year))

    ids = []
    for i, ch in enumerate(anchors):
        cid = f"C{i:02d}"
        ids.append(cid)
        start_ms = int(round(ch["start"] * 1000))
        end_ms = int(round((ch["start"] + ch["sec"]) * 1000))
        tags.add(CHAP(
            element_id=cid,
            start_time=start_ms,
            start_offset=None,
            end_time=end_ms,
            end_offset=None,
            sub_frames={},
        ))

    # The table of contents must point at the CHAP frames themselves.
    # Per ID3v2.4 a CTOC's child list may contain CHAP element ids directly.
    # An earlier version built one empty CTOC per chapter and listed *those*
    # as the TOC's children, which left every CHAP frame unreferenced - the
    # hierarchy was a dead end, and `read_chapters` (which only counts CHAP
    # frames) could not see it.
    tags.add(CTOC(
        element_id="TOC",
        flags=CTOC_TOP_LEVEL,
        child_element_ids=ids,
        sub_frames=[],
    ))
    tags.save(path, v2_version=4)


def read_chapters(path: str) -> list[tuple[str, int, int]]:
    """Read back (element_id, start_ms, end_ms) - used to verify the write."""
    tags = ID3(path)
    out = []
    for key in sorted(k for k in tags.keys() if k.startswith("CHAP:")):
        frame = tags[key]
        out.append((key.split(":", 1)[1], frame.start_time, frame.end_time))
    return out


def read_toc(path: str) -> tuple[str, list[str]]:
    """Read the top-level table of contents: (element_id, child element ids).

    Returns an empty tuple when the file has no top-level CTOC.
    """
    tags = ID3(path)
    fallback = ("", [])
    for key in sorted(k for k in tags.keys() if k.startswith("CTOC:")):
        frame = tags[key]
        children = list(getattr(frame, "child_element_ids", None) or [])
        if int(getattr(frame, "flags", 0) or 0) & CTOC_TOP_LEVEL:
            return key.split(":", 1)[1], children
        if len(children) > len(fallback[1]):
            fallback = (key.split(":", 1)[1], children)
    return fallback
