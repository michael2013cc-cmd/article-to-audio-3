#!/usr/bin/env python3
"""The Chinese numeral engine - the part every number check depends on."""

import unittest

from a2a.numbers import (
    numeric_forms,
    normalize_script,
    parse_chinese_number,
    renderings,
    scan_chinese_runs,
    to_chinese_number,
)


class TestParse(unittest.TestCase):
    def test_simple_integers(self):
        self.assertEqual(parse_chinese_number("二十九"), (29, None))
        self.assertEqual(parse_chinese_number("六百"), (600, None))
        self.assertEqual(parse_chinese_number("十"), (10, None))
        self.assertEqual(parse_chinese_number("三"), (3, None))

    def test_big_units(self):
        self.assertEqual(parse_chinese_number("三千亿"), (3e11, "亿"))
        self.assertEqual(parse_chinese_number("两万"), (20000, "万"))
        self.assertEqual(parse_chinese_number("二万五千"), (25000, None))

    def test_decimals(self):
        self.assertEqual(parse_chinese_number("一点五"), (1.5, None))
        self.assertEqual(parse_chinese_number("一百二十八点七"), (128.7, None))

    def test_percent(self):
        # 百分之八十六 is 86, and 百分之 is part of the numeral.
        self.assertEqual(parse_chinese_number("百分之八十六"), (86, None))

    def test_the_number_that_matters(self):
        # This is the figure from the test material. The listening script
        # spells it out, and the key is the Arabic form.
        self.assertEqual(parse_chinese_number("一万二千二百七十六"), (12276, None))

    def test_large_values_survive_float_rounding(self):
        # 294.4 亿 is 29440000000, not 29439999999.999996. The scaled
        # rendering is produced by renderings(), which is where the unit
        # is applied.
        value, unit = parse_chinese_number("二百九十四点四亿")
        self.assertEqual(value, 294.4)
        self.assertEqual(unit, "亿")
        self.assertIn("29440000000", renderings(value, unit))
        self.assertIn("29440000000", numeric_forms(29440000000))

    def test_malformed(self):
        self.assertIsNone(parse_chinese_number(""))


class TestRender(unittest.TestCase):
    def test_renderings_plain(self):
        self.assertEqual(renderings(12276), ["12276"])

    def test_renderings_scaled(self):
        forms = renderings(29440000000, "亿")
        self.assertIn("29440000000", forms)
        # The "same number in the other scale" alternative lives in
        # numeric_forms(); renderings() only scales an exact unit match.
        self.assertIn("2944000萬", numeric_forms(29440000000))

    def test_renderings_of_a_scaled_decimal(self):
        forms = renderings(294.4, "亿")
        self.assertIn("294.4", forms)
        self.assertIn("29440000000", forms)

    def test_to_chinese_number(self):
        self.assertEqual(to_chinese_number(2026), "二千零二十六")
        self.assertEqual(to_chinese_number(18), "十八")
        self.assertEqual(to_chinese_number(110), "一百一十")
        self.assertEqual(to_chinese_number(38000), "三万八千")

    def test_numeric_forms_covers_both_directions(self):
        # The STT turns Chinese numerals into Arabic AND Arabic numerals
        # into Chinese, so a key has to accept both.
        forms = numeric_forms(2026)
        self.assertIn("2026", forms)
        self.assertIn("二千零二十六", forms)
        # A year is spoken digit by digit, so that form has to be there too.
        self.assertIn("二零二六", forms)


class TestScan(unittest.TestCase):
    def test_finds_the_year(self):
        runs = scan_chinese_runs(list("二零二六年增加百分之八十六"))
        self.assertEqual([r["run"] for r in runs], ["二零二六", "百分之八十六"])

    def test_kilowatt_is_not_a_thousands_multiplier(self):
        # 一百六十七千瓦 is 167 kW. Absorbing the 千 produced 7160瓦,
        # wrong by a factor of ten.
        runs = scan_chinese_runs(list("一百六十七千瓦"))
        self.assertEqual([r["run"] for r in runs], ["一百六十七"])

    def test_bare_small_integers_are_words(self):
        # 一个 / 两次 are words, not figures to verify.
        self.assertEqual(scan_chinese_runs(list("一个两次")), [])

    def test_ordinals_are_labels(self):
        self.assertEqual(scan_chinese_runs(list("第三轮")), [])

    def test_vague_quantities_are_units(self):
        self.assertEqual(scan_chinese_runs(list("数百万用户")), [])

    def test_year_is_read_digit_by_digit(self):
        # 二零二六 is 2026. Read as a plain numeral it collapses to 6,
        # because each digit replaces the last one - and a value under ten
        # is then dropped as "a word, not a figure". The v1 implementation
        # had that hole: a script saying 二零二六年 never matched the key
        # 2026.
        runs = scan_chinese_runs(list("二零二六年"))
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["value"], 2026)

    def test_year_with_the_ideographic_zero(self):
        runs = scan_chinese_runs(list("二〇一五年"))
        self.assertEqual(runs[0]["value"], 2015)

    def test_a_plain_digit_run_is_not_a_year(self):
        # Without 年 in front, 一二三 stays a serial number.
        self.assertEqual(scan_chinese_runs(list("编号一二三")), [])


class TestNormalize(unittest.TestCase):
    def test_year_and_percentage(self):
        self.assertEqual(
            normalize_script("二零二六年增加百分之八十六"),
            "2026年增加86",
        )

    def test_leaves_arabic_alone(self):
        self.assertEqual(
            normalize_script("从12276增到19807"),
            "从12276增到19807",
        )

    def test_normalises_a_year(self):
        self.assertEqual(normalize_script("二零二六年"), "2026年")
        self.assertEqual(normalize_script("二〇一五年"), "2015年")


if __name__ == "__main__":
    unittest.main()
