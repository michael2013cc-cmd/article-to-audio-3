#!/usr/bin/env python3
"""Listen back and check the figures that were actually spoken.

Combines the local speech-to-text pass (:mod:`a2a.stt`) with the key
matching ported from v1 (:mod:`a2a.match`).
"""

from __future__ import annotations

import json
import os

from . import match, stt


def check_content(
    path: str,
    keys: list,
    model_size: str = stt.DEFAULT_MODEL,
    device: str = "auto",
    transcript_path: str | None = None,
    quiet: bool = False,
) -> dict:
    """Transcribe the finished audio and check every key against it.

    Returns a report with the transcript, the per-key verdict and the list
    of missing keys. ``ok`` is True only when every key was heard.
    """
    if not keys:
        return {
            "ok": False,
            "skipped": True,
            "reason": "no keys to check",
            "total": 0,
            "found": 0,
            "missing": [],
        }

    result = stt.transcribe(path, model_size=model_size, device=device, progress=not quiet)
    text = result["text"]
    if transcript_path:
        with open(transcript_path, "w", encoding="utf-8") as fh:
            fh.write(text)

    report = match.match_report(text, keys)
    report["transcript_path"] = transcript_path
    report["segments"] = result["segments"]
    report["transcript"] = text
    report["model"] = model_size
    if not quiet:
        print(
            f"      回听：{report['found']}/{report['total']} 个关键数字命中",
            flush=True,
        )
        for row in report["missing"]:
            print(f"      !! 没听到：{row['key']}", flush=True)
    return report


def load_keys(path: str) -> list:
    """Read a key file.

    Accepts a JSON list of strings or of ``{"key": ..., "alts": [...]}``
    objects, and also a JSON object of that shape under "keys".
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = data.get("keys", [])
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of keys")
    return data


def save_keys(path: str, keys: list) -> str:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(keys, fh, ensure_ascii=False, indent=2)
    return path
