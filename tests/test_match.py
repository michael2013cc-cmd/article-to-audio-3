#!/usr/bin/env python3
"""Key matching against a transcript - the three-layer comparison."""

import unittest

from a2a.match import match_keys, match_report


class TestMatch(unittest.TestCase):
    def test_plain_hit(self):
        self.assertEqual(match_keys("证文字符从12276增到19807", ["12276"]), [])

    def test_miss(self):
        self.assertEqual(match_keys("完全没有这个数", ["12276"]), ["12276"])

    def test_half_width_comma(self):
        # ep18: script 一万二千二百八十八个加速器 -> transcript 12,288
        self.assertEqual(match_keys("从 12,288 个加速器", ["12288"]), [])

    def test_full_width_comma(self):
        self.assertEqual(match_keys("从 1，180 个加速器", ["1180"]), [])

    def test_chinese_numeral_and_space(self):
        # ep16b: the transcript writes the first half of a thousands group
        # as a Chinese numeral followed by a space.
        self.assertEqual(match_keys("从一 180 个加速器", ["1180"]), [])

    def test_key_with_separator_matches_plain_transcript(self):
        # ep08a: the key carries the separator, the transcript does not.
        self.assertEqual(match_keys("四萬五千個加速器", ["45,000"]), [])

    def test_chinese_in_transcript_normalises(self):
        self.assertEqual(match_keys("增加百分之六十一", ["61"]), [])

    def test_object_keys_accept_any_alternative(self):
        keys = [{"key": "61", "alts": ["61", "六十一"]}]
        self.assertEqual(match_keys("增加六十一", keys), [])


class TestReport(unittest.TestCase):
    def test_report_lists_hits_and_misses(self):
        report = match_report("从12276增到19807", ["12276", "61"])
        self.assertEqual(report["found"], 1)
        self.assertEqual(report["total"], 2)
        self.assertFalse(report["ok"])
        self.assertEqual([r["key"] for r in report["missing"]], ["61"])

    def test_report_ok_when_everything_hit(self):
        report = match_report("从12276增到19807", ["12276", "19807"])
        self.assertTrue(report["ok"])
        self.assertEqual(report["found"], 2)


if __name__ == "__main__":
    unittest.main()
