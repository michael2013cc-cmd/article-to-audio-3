#!/usr/bin/env python3
"""The listen-back wrapper - tested without loading a real model.

The point of these tests is the call shape: faster-whisper iterates
clip_timestamps directly, so handing it None raises
"TypeError: 'NoneType' object is not iterable". That killed the first
real run of this pipeline, so it gets a test.
"""

import unittest

from a2a import stt


class FakeSegment:
    def __init__(self, start, end, text):
        self.start = start
        self.end = end
        self.text = text


class FakeInfo:
    language = "zh"
    duration = 12.5


class FakeModel:
    def __init__(self):
        self.kwargs = None

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        return (
            iter([
                FakeSegment(0.0, 1.0, "正文字符从"),
                FakeSegment(1.0, 2.0, " 12276 增到 19807。"),
                FakeSegment(2.0, 2.5, "   "),
            ]),
            FakeInfo(),
        )


class TestTranscribe(unittest.TestCase):
    def test_no_clip_timestamps_by_default(self):
        model = FakeModel()
        result = stt.transcribe("x.mp3", model=model)
        self.assertNotIn("clip_timestamps", model.kwargs)
        self.assertEqual(result["text"], "正文字符从12276 增到 19807。")
        self.assertEqual(len(result["segments"]), 2)

    def test_clip_timestamps_are_forwarded_when_wanted(self):
        model = FakeModel()
        stt.transcribe("x.mp3", model=model, clip_timestamps="108,200")
        self.assertEqual(model.kwargs["clip_timestamps"], "108,200")

    def test_repetition_loop_is_switched_off(self):
        # Whisper repeats itself into a loop on quiet audio, and the loop
        # text then looks like real content to a key matcher.
        model = FakeModel()
        stt.transcribe("x.mp3", model=model)
        self.assertFalse(model.kwargs["condition_on_previous_text"])


class TestDevice(unittest.TestCase):
    def test_cpu_is_always_available(self):
        self.assertEqual(stt.pick_device("cpu"), ("cpu", "int8"))

    def test_cuda_is_asked_for_explicitly(self):
        self.assertEqual(stt.pick_device("cuda"), ("cuda", "float16"))


if __name__ == "__main__":
    unittest.main()
