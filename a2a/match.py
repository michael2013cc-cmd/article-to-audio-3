#!/usr/bin/env python3
"""Matching verification keys against a transcript.

Ported from ``article-to-audio/scripts/match.mjs`` (v1), with the same
three-layer comparison:

  1. the raw text (tolerates whitespace and punctuation differences);
  2. the normalised text (Chinese numerals -> Arabic);
  3. the "tight" text (separators between digits removed).

Layer 3 exists because the STT writes thousands separators in THREE
different shapes, and the key never has any of them:

    "12,288"  half-width comma
    "1，180"  full-width comma
    "一 180"  Chinese numeral + a space

Measured on ep18: the script said "一万二千二百八十八个加速器", the
transcript said "12,288 个加速器", and the key was "12288". Without layer
3 that key was reported as missing audio.
"""

from __future__ import annotations

import re

from .numbers import normalize_script

# A single Chinese numeral ("一", "兩"). In the normaliser these are
# treated as words and skipped (a value < 10 with no big unit), because
# "一个" / "两次" are words, not figures. But the STT sometimes writes the
# first half of a thousands group as a Chinese numeral ("一 180" = 1,180),
# so this table rewrites it before the separators are squeezed out.
CN_DIGIT_1 = {
    "零": "0", "〇": "0", "一": "1", "二": "2", "兩": "2", "两": "2",
    "三": "3", "四": "4", "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}

_CN_PREFIX = re.compile(r"([零〇一二兩两三四五六七八九])[\s,，]+(?=\d)")
_DIGIT_SEP = re.compile(r"(\d)[\s,，]+(?=\d)")
_SEP_RUN = re.compile(r"[,，\s]+")


def tighten(text: str) -> tuple[str, str, str]:
    """Return ``(normalised, cn_normalised, tight)`` views of the text."""
    norm = normalize_script(text)
    norm_cn = _CN_PREFIX.sub(lambda m: f"{CN_DIGIT_1[m.group(1)]} ", norm)
    norm_tight = _DIGIT_SEP.sub(r"\1", norm_cn)
    return norm, norm_cn, norm_tight


def _key_forms(key) -> tuple[str, list[str]]:
    """The label of a key and every rendering that counts as a hit.

    A key file may hold plain strings instead of ``{"key": ..., "alts": [...]}``
    objects.
    """
    if isinstance(key, dict):
        label = str(key.get("key", ""))
        alts = key.get("alts")
        forms = [str(a) for a in alts] if isinstance(alts, list) and alts else [label]
    else:
        label = str(key)
        forms = [label]
    forms = forms + [label]

    # The key may carry thousands separators ("45,000") while the normalised
    # transcript never does: the TTS reads "45,000" as "四萬五千", which
    # normalises to "45000". Comparing the comma-bearing key against it
    # always misses, so the separator-free form is added as an accepted
    # rendering. Measured on ep08a: 45,000 / 30,000 / 25,000 all missed
    # this way.
    i = 0
    while i < len(forms):
        f = forms[i]
        plain = _SEP_RUN.sub("", f)
        if plain != f:
            forms.append(plain)
        i += 1
    return label, forms


def match_keys(text: str, keys: list) -> list:
    """Return the keys that are NOT found in ``text``."""
    norm, _cn, tight = tighten(text)
    missing = []
    for k in keys:
        _label, forms = _key_forms(k)
        if not any(f in text or f in norm or f in tight for f in forms):
            missing.append(k)
    return missing


def match_report(text: str, keys: list) -> dict:
    """Per-key verdict, not just the misses.

    ``found`` keys are reported too, because a verification that only
    says "29 of 29" leaves no way to tell which anchors were checked.
    """
    norm, _cn, tight = tighten(text)
    rows = []
    for k in keys:
        label, forms = _key_forms(k)
        hit = None
        for f in forms:
            if f in text:
                hit = f
                break
            if f in norm:
                hit = f
                break
            if f in tight:
                hit = f
                break
        rows.append({
            "key": label,
            "found": hit is not None,
            "matched_form": hit,
            "alts": forms,
            "source": k,
        })
    return {
        "total": len(rows),
        "found": sum(1 for r in rows if r["found"]),
        "missing": [r for r in rows if not r["found"]],
        "rows": rows,
        "ok": all(r["found"] for r in rows),
    }
