#!/usr/bin/env python3
"""Aligning the spoken words back to the script."""

import unittest

from a2a.align import align, sentence_spans, sentences


def mark(start, duration, text):
    return {"start": start, "duration": duration, "text": text}


class TestAlign(unittest.TestCase):
    def test_words_line_up_with_the_script(self):
        body = "先看硬指标，11个指标。"
        marks = [
            mark(0.1, 0.3, "先"),
            mark(0.4, 0.3, "看硬"),
            mark(0.8, 0.5, "指标"),
            mark(1.3, 0.4, "11个"),
            mark(1.7, 0.4, "指标"),
        ]
        aligned = align(body, marks)
        self.assertEqual([a["pos"] for a in aligned], [0, 1, 3, 6, 9])
        self.assertTrue(all(a["pos"] >= 0 for a in aligned))

    def test_unplaceable_word_is_marked(self):
        aligned = align("完全不同的句子", [mark(0.0, 0.1, "查无此词")])
        self.assertEqual(aligned[0]["pos"], -1)


class TestSpans(unittest.TestCase):
    def test_splits_on_sentence_marks(self):
        body = "第一句。第二句！第三句？"
        spans = sentence_spans(body)
        self.assertEqual(len(spans), 3)
        self.assertEqual(body[spans[0][0]:spans[0][1]], "第一句。")

    def test_keeps_the_last_fragment(self):
        body = "有句号。还有半句"
        spans = sentence_spans(body)
        self.assertEqual(len(spans), 2)


class TestSentences(unittest.TestCase):
    def test_each_sentence_gets_its_own_time(self):
        body = "正文字符从12276增到19807。增加61%。"
        marks = [
            mark(0.1, 0.2, "正"),
            mark(0.4, 0.3, "文字符"),
            mark(0.8, 0.2, "从"),
            mark(1.0, 0.6, "12276"),
            mark(1.7, 0.4, "增到"),
            mark(2.2, 0.6, "19807"),
            mark(3.0, 0.4, "增加"),
            mark(3.5, 0.5, "61%"),
        ]
        sents = sentences(body, marks)
        self.assertEqual(len(sents), 2)
        self.assertEqual(sents[0]["text"], "正文字符从12276增到19807。")
        self.assertAlmostEqual(sents[0]["start"], 0.1)
        self.assertAlmostEqual(sents[0]["end"], 2.8)
        self.assertEqual(sents[1]["text"], "增加61%。")
        self.assertAlmostEqual(sents[1]["start"], 3.0)


if __name__ == "__main__":
    unittest.main()
