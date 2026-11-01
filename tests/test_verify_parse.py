#!/usr/bin/env python3
"""Regression test for the ebur128 output parser.

The first version of this tool read ``I: -70.0`` off ffmpeg's per-frame
progress lines and declared a perfectly loud file silent. Real output shape
is reproduced here so the parser cannot regress without ffmpeg present.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2a.verify import _grab, _summary

RAW = """
[Parsed_ebur128_0 @ 0x5607] t: 0.5  TARGET:-23 LUFS M: -70.0 S: -70.0 I: -70.0 LUFS LRA: -nan LU Peak: -70.0 dB
[Parsed_ebur128_0 @ 0x5607] t: 60.2 TARGET:-23 LUFS M: -13.4 S: -13.1 I: -12.8 LUFS LRA: 2.2 LU Peak: -1.4 dB
[Parsed_ebur128_0 @ 0x5607] Summary:

  Integrated loudness:
    I:         -12.8 LUFS
    Threshold: -23.1 LUFS

  Loudness range:
    LRA:         2.2 LU
    Threshold: -33.2 LUFS
    LRA low:   -14.3 LUFS
    LRA high:  -12.0 LUFS

  True peak:
    Peak:       -1.4 dBFS
"""


class SummaryParseTest(unittest.TestCase):
    def test_only_the_summary_block_is_read(self):
        clean = _summary(RAW)
        self.assertFalse(clean.startswith("[Parsed_ebur128_0 @ 0x5607] t:"))
        self.assertEqual(_grab(clean, "I"), -12.8)
        self.assertEqual(_grab(clean, "LRA"), 2.2)
        self.assertEqual(_grab(clean, "Peak"), -1.4)

    def test_naive_parse_is_the_bug_being_guarded(self):
        self.assertEqual(_grab(RAW, "I"), -70.0)

    def test_missing_summary_falls_back_to_whole_text(self):
        self.assertEqual(_summary("no summary here"), "no summary here")

    def test_empty_output_does_not_crash_the_parser(self):
        # On a Windows console the ffmpeg output used to fail to decode, which
        # made subprocess return stderr=None and crashed _summary(None) with
        # an AttributeError instead of a readable report.
        self.assertEqual(_summary(""), "")
        self.assertEqual(_summary(None), "")


if __name__ == "__main__":
    unittest.main()
