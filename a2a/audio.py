#!/usr/bin/env python3
"""Loudness normalisation and lossless chapter concatenation (ffmpeg).

Why this exists: raw TTS output is quiet (around -18 LUFS) and can still clip
on phone speakers. Listening on a phone at maximum volume is a real failure
mode, so the chain here targets roughly -13 LUFS integrated with a true peak
safely below 0 dBFS.

The limiter ceiling is deliberately 0.76, not 0.80: MP3 encoding after the
limiter adds overshoot, and 0.80 was measured pushing the encoded file to
+0.28 dBFS, i.e. clipping. Verify with ``verify.check_file`` after any change.

Stereo output is intentional - mono sounds thin through a phone speaker.
"""

from __future__ import annotations

import os
import subprocess

# threshold/ratio/attack/release + makeup gain, a presence lift for
# intelligency, then a fixed +10 dB and a hard ceiling.
LOUDNESS_CHAIN = (
    "acompressor=threshold=-22dB:ratio=6:attack=5:release=200:makeup=12,"
    "equalizer=f=2800:t=q:w=1.4:g=3,"
    "volume=10dB,"
    "alimiter=limit=0.76:attack=3:release=150:level=disabled"
)

SAMPLE_RATE = 44100
BITRATE = "64k"  # transparent enough for speech, keeps an hour-long file small


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    # encoding must be explicit: without it Python decodes ffmpeg's output
    # with the locale code page, and a cp950 console cannot decode ffmpeg's
    # UTF-8 bytes -> UnicodeDecodeError -> stderr becomes None.
    proc = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd)}\n{proc.stderr[-1500:]}")
    return proc


def normalize(src: str, dst: str, chain: str = LOUDNESS_CHAIN,
              bitrate: str = BITRATE) -> str:
    """Re-encode one chapter to the target loudness, stereo, fixed bitrate."""
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", src,
        "-af", chain,
        "-ar", str(SAMPLE_RATE), "-ac", "2", "-b:a", bitrate,
        dst,
    ])
    return dst


def normalize_chapters(raw_paths: list[str], workdir: str,
                       chain: str = LOUDNESS_CHAIN,
                       bitrate: str = BITRATE) -> list[str]:
    out = []
    for i, src in enumerate(raw_paths):
        dst = os.path.join(workdir, f"norm_{i}.mp3")
        normalize(src, dst, chain, bitrate)
        out.append(dst)
    return out


def concat(parts: list[str], dst: str, workdir: str) -> str:
    """Concatenate chapter mp3s with no re-encode (stream copy).

    All parts must share codec, sample rate and channel count - normalize()
    guarantees that. ``-write_xing`` fixes the duration header, otherwise
    players show a wrong length.
    """
    listfile = os.path.join(workdir, "concat_list.txt")
    with open(listfile, "w", encoding="utf-8") as fh:
        for p in parts:
            # single quotes inside a concat manifest are escaped, not quoted
            fh.write("file '" + os.path.abspath(p).replace("'", "'\\''") + "'\n")
    _run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "concat", "-safe", "0", "-i", listfile,
        "-c", "copy", "-write_xing", "1",
        dst,
    ])
    return dst


def duration(path: str) -> float:
    """Duration in seconds, read from the file itself, not from estimates.

    Chapter anchors must come from the *normalized* parts, because the
    limiter and the encoder change the length slightly.
    """
    from mutagen.mp3 import MP3

    return MP3(path).info.length
