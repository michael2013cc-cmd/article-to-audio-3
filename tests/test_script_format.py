#!/usr/bin/env python3
"""Format contract tests - no network, no ffmpeg. ``python -m unittest discover tests``"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2a import script
from a2a.transcript import build_transcript, clock

GOOD = """@@P0|第一节
第一段内容。

第二段内容。

@@P1|第二节
只有这一段。
"""


class ParseTest(unittest.TestCase):
    def test_two_chapters(self):
        chapters = script.parse(GOOD)
        self.assertEqual(len(chapters), 2)
        self.assertEqual([c.id for c in chapters], ["P0", "P1"])
        self.assertEqual([c.title for c in chapters], ["第一节", "第二节"])
        self.assertIn("第二段内容", chapters[0].body)
        self.assertNotIn("\n\n\n", chapters[0].body)

    def test_round_trip(self):
        a = script.parse(GOOD)
        b = script.parse(script.dump(a))
        self.assertEqual([(c.id, c.title, c.body) for c in a],
                         [(c.id, c.title, c.body) for c in b])

    def test_marker_without_pipe_is_fatal(self):
        with self.assertRaises(ValueError):
            script.parse("@@P0 缺了竖线\n正文\n")

    def test_body_before_first_marker_is_fatal(self):
        with self.assertRaises(ValueError):
            script.parse("前言写在这里了\n@@P0|第一节\n正文\n")

    def test_no_markers_is_fatal(self):
        with self.assertRaises(ValueError):
            script.parse("整篇都没有章节标记\n")

    def test_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.txt")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(GOOD)
            self.assertEqual(len(script.load(path)), 2)


class TranscriptTest(unittest.TestCase):
    def test_clock(self):
        self.assertEqual(clock(0), "00:00")
        self.assertEqual(clock(90), "01:30")
        self.assertEqual(clock(3599.6), "60:00")

    def test_anchors_and_body(self):
        chapters = script.parse(GOOD)
        anchors = [
            {"start": 0.0, "sec": 61.0, "title": "第一节"},
            {"start": 61.0, "sec": 30.0, "title": "第二节"},
        ]
        text = build_transcript(chapters, anchors, audio_name="x.mp3", title="样音")
        self.assertIn("01:01  第2节", text)
        self.assertIn("【01:01】第2节", text)
        self.assertIn("总时长：01:31", text)
        self.assertEqual(text.count("第二节"), 2)  # 时间点索引 + 小节标题


if __name__ == "__main__":
    unittest.main()
