#!/usr/bin/env python3
"""Measure the finished file: loudness, clipping, decodability, chapter count.

Numbers to aim for, learned the hard way from a file that measured -18.7 LUFS
but had a true peak of +2.8 dBFS - quiet *and* distorted at the same time:

  integrated loudness  ~ -13 LUFS   (loud enough on a phone)
  true peak            <= -0.2 dBFS (never above 0)
  flat factor          0.0          (sustained digital silence = clipping)
  decode               no errors
"""

from __future__ import annotations

import re
import subprocess

TARGET_LUFS = (-16.0, -11.0)
MAX_TRUE_PEAK = -0.1
MAX_FLAT_FACTOR = 0.0


def _ff(args: list[str]) -> str:
    # encoding must be explicit - see audio._run. Without it the locale code
    # page decodes ffmpeg's UTF-8 output and the process dies inside the
    # reader thread, which surfaces as `stderr is None` further down.
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", *args, "-f", "null", "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.stderr or ""


def _grab(text: str, label: str) -> float | None:
    m = re.search(rf"{label}:\s*(-?[\d.]+)", text)
    return float(m.group(1)) if m else None


def _summary(text: str) -> str:
    """Keep only ffmpeg's trailing Summary block.

    ebur128 also prints running numbers for every analysed frame, and those
    read ``I: -70.0`` until there is enough audio to gate. Parsing the whole
    stderr therefore reports -70 LUFS for a perfectly normal file.
    """
    if not text:
        return ""
    idx = text.rfind("Summary:")
    return text[idx:] if idx != -1 else text


def loudness(path: str) -> dict:
    out = _summary(_ff(["-i", path, "-af", "ebur128=peak=true"]))
    return {
        "integrated_lufs": _grab(out, "I"),
        "lra": _grab(out, "LRA"),
        "true_peak_dbfs": _grab(out, "Peak"),
    }


def flat_factor(path: str) -> float | None:
    out = _ff(["-i", path, "-af", "astats=measure_perchannel=none"])
    return _grab(out, "Flat factor")


def decodes(path: str) -> bool:
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-v", "error", "-i", path, "-f", "null", "-"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return proc.returncode == 0 and not (proc.stderr or "").strip()


def check_file(path: str, expect_chapters: int | None = None) -> dict:
    """Return a measurement report plus a pass/fail verdict with reasons."""
    from .tags import read_chapters, read_toc

    report: dict = {"file": path}
    problems: list[str] = []

    loud = loudness(path)
    report.update(loud)
    lufs = loud["integrated_lufs"]
    peak = loud["true_peak_dbfs"]
    if lufs is None or not (TARGET_LUFS[0] <= lufs <= TARGET_LUFS[1]):
        problems.append(f"integrated loudness {lufs} outside {TARGET_LUFS}")
    if peak is None or peak > MAX_TRUE_PEAK:
        problems.append(f"true peak {peak} dBFS above {MAX_TRUE_PEAK}")

    ff = flat_factor(path)
    report["flat_factor"] = ff
    if ff is None or ff > MAX_FLAT_FACTOR:
        problems.append(f"flat factor {ff} indicates possible clipping")

    report["decodes"] = decodes(path)
    if not report["decodes"]:
        problems.append("file does not decode cleanly")

    chapters = read_chapters(path)
    report["chapters"] = len(chapters)
    if expect_chapters is not None and len(chapters) != expect_chapters:
        problems.append(
            f"{len(chapters)} chapter frames, expected {expect_chapters}"
        )

    # The chapter table of contents has to actually point at the CHAP frames.
    # Counting CHAP frames alone cannot see a broken hierarchy.
    toc_id, toc_children = read_toc(path)
    report["toc_children"] = len(toc_children)
    chap_ids = {cid for cid, _, _ in chapters}
    if chap_ids:
        if not toc_children:
            problems.append("no top-level table of contents")
        elif not set(toc_children) <= chap_ids:
            problems.append("table of contents references unknown element ids")
        elif set(toc_children) != chap_ids:
            problems.append("table of contents does not list every chapter")

    report["ok"] = not problems
    report["problems"] = problems
    return report
