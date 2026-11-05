#!/usr/bin/env python3
"""The recheck: a miss is not a verdict until the window has been re-read.

All tests inject a scripted ``transcribe`` so the decision table runs
without a model. The table itself is the v1 lesson: distinguish
"recognition noise" (second read finds the figure -> fine), "damaged
audio" (both reads agree on the SAME other number -> re-render) and
"inconclusive" (tell the human where to listen).
"""

import unittest

from a2a import content


def _first_pass(segments, text=None):
    return {
        "text": text if text is not None else "".join(s["text"] for s in segments),
        "segments": segments,
        "duration": 40.0,
    }


class ScriptedTranscribe:
    """Full pass first, then scripted re-reads per clip window."""

    def __init__(self, first, windows=None, second_full=None):
        self.first = first
        self.windows = dict(windows or {})
        self.second_full = second_full
        self.calls = []

    def __call__(self, path, *, clip_timestamps=None, beam_size=5):
        self.calls.append((clip_timestamps, beam_size))
        if clip_timestamps is None:
            if beam_size == 1 and self.second_full is not None:
                return self.second_full
            return self.first
        return {"text": self.windows.get(clip_timestamps, ""),
                "segments": [], "duration": 40.0}


SENTENCES = [
    {"text": "正文字符从12276增到19807。", "start": 12.0, "end": 15.5,
     "chapter": 2, "chapter_title": "第三章"},
]
KEYS = [{"key": "12276", "alts": ["12276"]}]
WRONG = [{"start": 12.0, "end": 15.5, "text": "从11111增到"}]


class TestNoKeys(unittest.TestCase):
    def test_skips_everything(self):
        report = content.check_content("x.mp3", [], transcribe=lambda *a, **k: {})
        self.assertTrue(report["skipped"])
        self.assertFalse(report["ok"])


class TestRecheck(unittest.TestCase):
    def test_all_hit_means_no_second_pass(self):
        fake = ScriptedTranscribe(
            _first_pass([{"start": 12.0, "end": 15.5, "text": "从12276增到"}]))
        report = content.check_content(
            "x.mp3", KEYS, transcribe=fake, sentences=SENTENCES, quiet=True)
        self.assertTrue(report["ok"])
        self.assertEqual(len(fake.calls), 1)

    def test_miss_recovered_in_window(self):
        fake = ScriptedTranscribe(
            _first_pass(WRONG), {"12.0,15.5": "从12276增到"})
        report = content.check_content(
            "x.mp3", KEYS, transcribe=fake, sentences=SENTENCES, quiet=True)
        self.assertTrue(report["ok"])
        self.assertEqual(report["recheck"]["recovered"], ["12276"])
        self.assertEqual(report["missing"], [])
        # the re-read is zoomed into the sentence and differently configured
        self.assertIn(("12.0,15.5", 1), fake.calls)

    def test_miss_agreeing_on_the_same_wrong_value_is_suspect_audio(self):
        fake = ScriptedTranscribe(
            _first_pass(WRONG), {"12.0,15.5": "从11111增到"})
        report = content.check_content(
            "x.mp3", KEYS, transcribe=fake, sentences=SENTENCES, quiet=True)
        self.assertFalse(report["ok"])
        self.assertEqual(report["recheck"]["suspect_audio"], ["12276"])
        row = report["missing"][0]
        self.assertEqual(row["verdict"], "suspect_audio")
        self.assertEqual(row["evidence"], "11111")
        self.assertEqual(row["chapter"], 2)
        self.assertEqual(row["chapter_title"], "第三章")

    def test_miss_without_agreement_is_inconclusive(self):
        fake = ScriptedTranscribe(
            _first_pass(WRONG), {"12.0,15.5": "从某个数增到"})
        report = content.check_content(
            "x.mp3", KEYS, transcribe=fake, sentences=SENTENCES, quiet=True)
        self.assertFalse(report["ok"])
        self.assertEqual(report["recheck"]["inconclusive"], ["12276"])
        row = report["missing"][0]
        self.assertEqual(row["verdict"], "inconclusive")
        # the human is told WHERE to listen, not just that it missed
        self.assertEqual(row["window"], (12.0, 15.5))

    def test_recheck_disabled(self):
        fake = ScriptedTranscribe(_first_pass(WRONG))
        report = content.check_content(
            "x.mp3", KEYS, transcribe=fake, sentences=SENTENCES,
            recheck=False, quiet=True)
        self.assertFalse(report["ok"])
        self.assertEqual(len(fake.calls), 1)
        self.assertNotIn("recheck", report)


class TestRecheckWithoutSentences(unittest.TestCase):
    def test_full_second_pass_recovers(self):
        fake = ScriptedTranscribe(
            _first_pass(WRONG),
            second_full=_first_pass([], text="从12276增到"),
        )
        report = content.check_content(
            "x.mp3", KEYS, transcribe=fake, quiet=True)
        self.assertTrue(report["ok"])
        self.assertEqual(report["recheck"]["recovered"], ["12276"])

    def test_full_second_pass_still_missing_is_inconclusive(self):
        # No window -> no way to localise an "agreed wrong value", so the
        # honest verdict is inconclusive, never a guess.
        fake = ScriptedTranscribe(_first_pass(WRONG))
        report = content.check_content(
            "x.mp3", KEYS, transcribe=fake, quiet=True)
        self.assertFalse(report["ok"])
        self.assertEqual(report["recheck"]["inconclusive"], ["12276"])


if __name__ == "__main__":
    unittest.main()
