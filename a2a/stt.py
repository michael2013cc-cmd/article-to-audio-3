#!/usr/bin/env python3
"""Listen back to the finished audio and check what was actually said.

This is the feature v1 had and v2 did not: after the audio exists, run a
speech-to-text pass over it and compare the result against a set of
verification keys. If a figure is wrong in the audio, this is where it
shows up.

It runs on a local model (faster-whisper), so it costs nothing and needs
no account, no key and no network. The model is downloaded once.

WHY ONLY THE NUMBERS ARE CHECKED
Measured on this machine, Whisper transcribes Chinese numerals back into
Arabic ("增加百分之六十一" -> "增加61%"), so a key written as an Arabic
figure can be matched against a script that spells the number out. But the
surrounding words are not reliable: the same run turned "硬指标" into
"应指标" and "独立来源" into "不利来源", and large-v3 turned it into
"每签字来源".

Numbers survive; homophones do not. So the verification is deliberately
about figures and terms, not about a verbatim transcript.

MODEL SIZES, MEASURED HERE (CPU, int8, on the test material):

    small     0.5 GB   3.1x realtime   every figure correct
    medium    1.5 GB   ~1.2x realtime  figures correct, more homophone slips
    large-v3  3.0 GB   0.64x realtime  figures correct, still homophone slips

The default is ``large-v3``: it is the most accurate of the three, and the
figures it reads back are the ones this pipeline checks. On a CPU it is
slower than real time, so a 26-minute file takes about 40 minutes to
verify - which is why the CUDA path matters: with the runtime libraries
installed, the same model finishes in a couple of minutes on a GPU.

Pass ``--stt-model small`` for a quick check that does not need a GPU.
"""

from __future__ import annotations

import os
import sys

DEFAULT_MODEL = "large-v3"
DEFAULT_LANGUAGE = "zh"


class SttUnavailable(RuntimeError):
    """Raised when the local speech-to-text stack is not installed."""


INSTALL_HINT = (
    "回聽驗證需要 faster-whisper：\n"
    "    pip install faster-whisper\n"
    "（第一次執行會下載模型，large-v3 約 3 GB；想快一點可用 --stt-model small。）"
)


def _import_whisper():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise SttUnavailable(INSTALL_HINT) from exc
    return WhisperModel


def _cuda_dll_dirs() -> list[str]:
    """Directories that hold the CUDA runtime libraries ctranslate2 needs.

    The pip packages ``nvidia-cublas-cu12`` and ``nvidia-cudnn-cu12`` put
    their DLLs in ``site-packages/nvidia/<package>/bin``. Windows will not
    find them on its own - they have to be registered with
    ``os.add_dll_directory`` or put on PATH.
    """
    dirs: list[str] = []
    try:
        import site
        bases = list(site.getsitepackages()) + [site.getusersitepackages()]
    except Exception:  # pragma: no cover - site layout varies
        bases = []
    for base in bases:
        nvidia = os.path.join(base, "nvidia")
        if not os.path.isdir(nvidia):
            continue
        for entry in sorted(os.listdir(nvidia)):
            candidate = os.path.join(nvidia, entry, "bin")
            if os.path.isdir(candidate):
                dirs.append(candidate)
    return dirs


def _prepare_cuda_dlls() -> bool:
    """Make the CUDA libraries loadable. Returns True if any were found."""
    dirs = _cuda_dll_dirs()
    if not dirs:
        return False
    for folder in dirs:
        try:
            os.add_dll_directory(folder)
        except (AttributeError, OSError):  # pragma: no cover - not on Windows
            pass
    os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])
    return True


def cuda_available() -> bool:
    """True when a CUDA device can actually be used."""
    try:
        import ctranslate2
    except ImportError:
        return False
    try:
        if ctranslate2.get_cuda_device_count() < 1:
            return False
    except Exception:
        return False
    _prepare_cuda_dlls()
    return True


def pick_device(device: str = "auto") -> tuple[str, str]:
    """Return ``(device, compute_type)`` for the requested device."""
    if device == "cuda":
        return "cuda", "float16"
    if device == "auto":
        if cuda_available():
            return "cuda", "float16"
        return "cpu", "int8"
    return device, "int8"


def load_model(model_size: str = DEFAULT_MODEL, device: str = "auto"):
    """Load a faster-whisper model. Returns ``(model, device, compute_type)``."""
    WhisperModel = _import_whisper()
    dev, compute = pick_device(device)
    try:
        return WhisperModel(model_size, device=dev, compute_type=compute), dev, compute
    except Exception as exc:
        if dev != "cuda":
            raise
        print(
            f"  CUDA 不可用（{exc!r}），改用 CPU int8。",
            file=sys.stderr,
            flush=True,
        )
        return WhisperModel(model_size, device="cpu", compute_type="int8"), "cpu", "int8"


def transcribe(
    path: str,
    model_size: str = DEFAULT_MODEL,
    device: str = "auto",
    language: str = DEFAULT_LANGUAGE,
    beam_size: int = 5,
    vad_filter: bool = True,
    clip_timestamps=None,
    model=None,
    progress: bool = True,
) -> dict:
    """Transcribe an audio file.

    Returns ``{"text": str, "segments": [{"start","end","text"}], ...}``.
    """
    if model is None:
        model, dev, compute = load_model(model_size, device)
        if progress:
            print(f"  model {model_size} on {dev}/{compute}", flush=True)

    # faster-whisper iterates clip_timestamps directly, so passing None
    # raises "TypeError: 'NoneType' object is not iterable". The option is
    # only added when it is actually wanted.
    options = {
        "language": language,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        # Whisper loves to repeat itself into a loop on quiet audio, and
        # the loop text then looks like real content to a key matcher.
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
    }
    if clip_timestamps is not None:
        options["clip_timestamps"] = clip_timestamps

    segments, info = model.transcribe(path, **options)

    rows = []
    chunks = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        rows.append({"start": float(s.start), "end": float(s.end), "text": text})
        chunks.append(text)
    return {
        "text": "".join(chunks),
        "segments": rows,
        "language": getattr(info, "language", language),
        "duration": float(getattr(info, "duration", 0.0) or 0.0),
    }


def transcribe_text(path: str, **kwargs) -> str:
    """Just the transcript text."""
    return transcribe(path, **kwargs)["text"]
