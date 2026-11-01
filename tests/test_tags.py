#!/usr/bin/env python3
"""Chapter table structure tests.

The chapter table of contents has to actually point at the CHAP frames. An
earlier version wrote one empty CTOC per chapter and listed *those* as the
TOC's children, so every CHAP frame was unreferenced - the hierarchy was a
dead end. Counting CHAP frames (what ``read_chapters`` does) cannot see that,
so it is asserted here.

Needs mutagen, not ffmpeg.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mutagen.id3 import ID3

from a2a import tags


def _write_dummy_mp3(path: str) -> None:
    """A file with an audio-ish payload; mutagen only needs it to exist."""
    with open(path, "wb") as fh:
        fh.write(b"\xff\xfb\x90\x00" + b"\x00" * 512)


ANCHORS = [
    {"start": 0.0, "sec": 10.5, "title": "开场"},
    {"start": 10.5, "sec": 20.25, "title": "证据"},
    {"start": 30.75, "sec": 5.0, "title": "收尾"},
]


class ChapterTableTest(unittest.TestCase):
    def _tagged(self, d: str) -> str:
        path = os.path.join(d, "x.mp3")
        _write_dummy_mp3(path)
        tags.write_chapters(path, ANCHORS, title="T", artist="A")
        return path

    def test_toc_children_are_the_chapter_frames(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._tagged(d)
            chapters = tags.read_chapters(path)
            toc_id, children = tags.read_toc(path)
            self.assertEqual(toc_id, "TOC")
            self.assertEqual([c[0] for c in chapters], children)

    def test_no_orphan_ctoc_frames(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._tagged(d)
            frames = ID3(path)
            ctocs = sorted(k for k in frames.keys() if k.startswith("CTOC:"))
            self.assertEqual(ctocs, ["CTOC:TOC"])

    def test_chapter_times_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._tagged(d)
            chapters = tags.read_chapters(path)
            self.assertEqual(chapters[0][1], 0)
            self.assertEqual(chapters[0][2], 10500)
            self.assertEqual(chapters[1][1], 10500)
            self.assertEqual(chapters[2][2], 35750)

    def test_top_level_ctoc_is_marked_as_top(self):
        with tempfile.TemporaryDirectory() as d:
            path = self._tagged(d)
            frames = ID3(path)
            flags = int(frames["CTOC:TOC"].flags)
            self.assertTrue(flags & tags.CTOC_TOP_LEVEL)

    def test_toc_children_are_not_empty(self):
        # The old writer passed `child=` and `is_top=`, which mutagen ignored,
        # so the TOC was written with an empty child list.
        with tempfile.TemporaryDirectory() as d:
            path = self._tagged(d)
            frames = ID3(path)
            self.assertEqual(list(frames["CTOC:TOC"].child_element_ids),
                             ["C00", "C01", "C02"])

    def test_each_chapter_carries_its_title(self):
        # A CHAP frame without a TIT2 sub-frame is still a valid chapter,
        # but a player that builds its chapter list from the CHAP frames
        # then shows entries with no names.
        with tempfile.TemporaryDirectory() as d:
            path = self._tagged(d)
            frames = ID3(path)
            titles = []
            for key in sorted(k for k in frames.keys() if k.startswith("CHAP:")):
                sub = frames[key].sub_frames
                self.assertIn("TIT2", sub, f"{key} has no title")
                titles.append(sub["TIT2"].text[0])
            self.assertEqual(titles, ["开场", "证据", "收尾"])


if __name__ == "__main__":
    unittest.main()
