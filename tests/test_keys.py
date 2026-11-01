#!/usr/bin/env python3
"""Key extraction, ranking and pre-flight."""

import unittest

from a2a.keys import extract_candidates, lint_numbers, preflight, rank_candidates


class TestExtract(unittest.TestCase):
    def test_tier_a_arabic(self):
        cands = extract_candidates(["正文字符从12276增到19807，增加61%。"])
        keys = {c["key"] for c in cands}
        self.assertIn("12276", keys)
        self.assertIn("19807", keys)
        self.assertIn("61", keys)
        self.assertTrue(all(c["tier"] == "A" for c in cands))

    def test_tier_b_chinese(self):
        cands = extract_candidates(["增加百分之六十一。"])
        by_key = {c["key"]: c for c in cands}
        self.assertIn("61", by_key)
        self.assertEqual(by_key["61"]["tier"], "B")
        self.assertEqual(by_key["61"]["cn_text"], "百分之六十一")

    def test_thousands_separator_is_one_number(self):
        cands = extract_candidates(["共 46,225 家机构。"])
        keys = [c["key"] for c in cands]
        self.assertIn("46,225", keys)
        self.assertNotIn("46", keys)

    def test_decimal_after_comma_group(self):
        cands = extract_candidates(["增速 7,065.6 万元。"])
        keys = [c["key"] for c in cands]
        self.assertIn("7,065.6", keys)

    def test_unit_is_part_of_the_alternatives(self):
        cands = extract_candidates(["增加 61%。"])
        by_key = {c["key"]: c for c in cands}
        self.assertTrue(by_key["61"]["has_unit"])


class TestRank(unittest.TestCase):
    def test_prefers_distinctive_figures(self):
        cands = extract_candidates([
            "正文字符从12276增到19807。",
            "增加61%。",
            "3个方案。",
        ])
        picked = rank_candidates(cands, top_n=3)
        self.assertEqual(len(picked), 3)
        self.assertNotIn("3", [c["key"] for c in picked])

    def test_spreads_across_segments(self):
        cands = extract_candidates([
            "12276 19807 61 93。",
            "147 1754 24454 10968。",
        ])
        picked = rank_candidates(cands, top_n=4)
        segments = {c["segment"] for c in picked}
        self.assertEqual(len(segments), 2)


class TestPreflight(unittest.TestCase):
    def test_bogus_keys_are_rejected(self):
        lines = ["正文字符从12276增到19807。"]
        bogus = preflight(["12276", "999"], lines)
        self.assertEqual(bogus, ["999"])

    def test_chinese_key_finds_its_arabic_form(self):
        lines = ["增加百分之六十一。"]
        self.assertEqual(preflight(["61"], lines), [])

    def test_year_key_finds_the_spelled_out_year(self):
        # The script spells the year out, the key is the Arabic form.
        lines = ["二零二六年的第二輪全面領先。"]
        self.assertEqual(preflight(["2026"], lines), [])

    def test_object_keys_use_their_alternatives(self):
        lines = ["增加百分之六十一。"]
        self.assertEqual(
            preflight([{"key": "61", "alts": ["61", "六十一"]}], lines),
            [],
        )


class TestLint(unittest.TestCase):
    def test_flags_chinese_decimals(self):
        warns = lint_numbers(["一点五倍。"])
        self.assertTrue(warns)
        self.assertIn("小数", warns[0]["reason"])

    def test_flags_ambiguous_scales(self):
        warns = lint_numbers(["三百万亿。"])
        self.assertTrue(warns)
        self.assertTrue(warns[0]["ambiguous"])


if __name__ == "__main__":
    unittest.main()
