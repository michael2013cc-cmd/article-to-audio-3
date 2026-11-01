#!/usr/bin/env python3
"""Mechanical audit of a listening script before it becomes audio.

Why this exists: a spoken rewrite of a document goes through a lot of manual
editing, and the two failure modes that actually happened were (a) a year
transcribed into the wrong Chinese numerals and (b) stray tokens from another
language leaking into the draft. Both are invisible while reading and
obvious while listening.

This module does not judge correctness - it extracts the number inventory and
the suspicious token list so a human can diff them against the source in
minutes instead of re-reading both files.
"""

from __future__ import annotations

import re
from collections import Counter

# 二零二六年, 二〇一五年 ... spelled-out years, the usual form in a spoken script.
# 〇 (U+3007, the ideographic zero) belongs in this class: it is a standard way
# to write a year, and leaving it out made "二〇一五年" invisible to the whole
# audit - a wrong year written that way slipped through unmatched_numbers.
CN_YEAR = re.compile(r"[零〇一二三四五六七八九十百]{3,5}年")
# 2026年, 1.5倍, 43.8%, 20/320
DIGITS = re.compile(r"\d+(?:\.\d+)?(?:%|倍|家|亿|万)?")
# any run containing Latin or Cyrillic letters
LATIN_RUN = re.compile(r"[A-Za-z\u0400-\u04FF][A-Za-z\u0400-\u04FF0-9'’\-]*")

# terms that legitimately appear in a Chinese tech script; everything else
# in LATIN_RUN is worth a look.
#
# The list used to be short, and every product name in a real document
# became a false positive: a single research script produced 25 flags, all
# of them brand names (TalkBack, Play Store, iPhone, Gemini...). The list
# below covers the usual tech vocabulary; anything else can be added per
# run with `audit --allow NAME`.
ALLOWED_LATIN = {
    # formats and units
    "AI", "App", "MP3", "MP4", "ID3", "PDF", "TTS", "STT", "API", "GB", "MB",
    "KB", "TB", "Hz", "kHz", "MHz", "GHz", "logMAR", "LUFS", "dBFS", "dB",
    "mm", "cm", "km", "kg", "ml", "pH", "FPS", "px", "K", "M",
    # hardware and platforms
    "VR", "AR", "MR", "XR", "CPU", "GPU", "NPU", "TPU", "OS", "PC", "USB",
    "HDMI", "Wi-Fi", "WiFi", "Bluetooth", "LTE", "5G", "4G", "IoT", "SSD",
    "DRAM", "RAM", "ROM", "LCD", "OLED", "LED", "DNS", "TCP", "HTTP", "URL",
    # software and services
    "Windows", "macOS", "Linux", "Android", "iOS", "iPadOS", "Chrome",
    "Safari", "Firefox", "Edge", "VSCode", "Git", "GitHub", "Python",
    "Node", "npm", "Docker", "Kubernetes", "Cloud", "AWS", "Azure", "GCP",
    "LLM", "RAG", "AGI", "GPU", "RLHF", "SFT", "API", "SDK", "IDE", "CLI",
    # accessibility and products
    "TalkBack", "VoiceOver", "NVDA", "JAWS", "Play", "Store", "iPhone",
    "iPad", "Mac", "Windows", "Android", "Gemini", "ChatGPT", "Claude",
    "Copilot", "Whisper", "TTS", "LUFS", "Lighthouse", "Playwright",
    "Axure", "Figma", "Zoom", "Slack", "Notion", "Excel", "PowerPoint",
    "Word", "Outlook", "Teams", "AirPods", "Siri", "Alexa", "HomeKit",
    # research and standards
    "WCAG", "ADA", "ARIA", "PDF", "ISO", "ANSI", "IEEE", "NIST", "UN",
    "UNESCO", "WHO", "NGO", "GDP", "IPO", "B2B", "B2C", "SaaS",
}


def spoken_text(text: str) -> str:
    """Drop marker lines: ``@@P0|标题`` is never read aloud, so its Latin
    letters and digits must not be reported as stray tokens."""
    return "\n".join(
        line for line in text.splitlines()
        if not line.startswith("@@")
    )


def spelled_years(text: str) -> Counter:
    """Count spelled-out years; each should match a year in the source."""
    return Counter(CN_YEAR.findall(text))


def digits(text: str) -> Counter:
    """Count numeric tokens in Arabic form."""
    return Counter(DIGITS.findall(text))


def latin_runs(text: str, allow: list[str] | None = None) -> Counter:
    """Every Latin/Cyrillic-looking run, lowercased for grouping."""
    runs = LATIN_RUN.findall(text)
    allowed = set(ALLOWED_LATIN)
    for extra in allow or []:
        allowed.add(extra)
        allowed.add(extra.lower())
    return Counter(r for r in runs if r not in allowed and r.lower() not in allowed)


# Unicode ranges that legitimately appear in a Chinese listening script
EXPECTED_RANGES = [
    (0x0000, 0x007F),  # ASCII (Latin letters are covered by latin_runs instead)
    (0x2000, 0x206F),  # general punctuation: dashes, curly quotes, ellipsis
    (0x2E80, 0x2FFF),  # CJK radicals and kangxi
    (0x3000, 0x303F),  # CJK punctuation
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified
    (0xF900, 0xFAFF),  # CJK compatibility
    (0xFF00, 0xFFEF),  # fullwidth forms
]


def _expected(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in EXPECTED_RANGES)


def non_cjk_characters(text: str) -> set[str]:
    """Characters from scripts a Chinese script should not contain.

    Catches the failure that actually happened: a Cyrillic fragment pasted in
    from a foreign source. Latin letters are excluded here because they are
    also proper nouns; ``latin_runs`` handles them with a whitelist.
    """
    return {ch for ch in text if not _expected(ch)}


CN_DIGITS = {
    "零": "0", "〇": "0", "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}


def cn_year_to_digits(token: str) -> str | None:
    """``二零一五年`` -> ``2015``; ``十六年`` -> None (not digit-by-digit)."""
    body = token.rstrip("年")
    out = "".join(CN_DIGITS.get(ch, "") for ch in body)
    return out if out and len(out) == len(body) else None


def _number_set(text: str) -> set[str]:
    """Arabic numbers plus spelled-out years, all as digits, for diffing."""
    numbers = set(DIGITS.findall(text))
    for token in CN_YEAR.findall(text):
        digits_form = cn_year_to_digits(token)
        if digits_form:
            numbers.add(digits_form)
    return numbers


def diff_against_source(script_text: str, source_text: str) -> dict:
    """Numeric tokens present in the script but absent from the source.

    These are the ones to double-check by hand: they are either derived
    (a ratio computed from two source numbers) or wrong. Spelled-out years
    are folded in, because 二零二五 vs 2015 in the source is exactly the
    transcription error this check exists to catch.
    """
    src = _number_set(source_text)
    script_numbers = _number_set(script_text)
    return {
        "unmatched_numbers": sorted(script_numbers - src),
        "source_only_numbers": sorted(src - script_numbers),
    }


def audit(script_path: str, source_path: str | None = None,
          allow: list[str] | None = None) -> dict:
    """Return the audit report. ``ok`` is a suggestion, not a verdict."""
    with open(script_path, encoding="utf-8") as fh:
        text = fh.read()
    spoken = spoken_text(text)

    years = spelled_years(spoken)
    suspicious = latin_runs(spoken, allow)
    report: dict = {
        "chars": len(text),
        "chapters": len(re.findall(r"^@@[^|\s]*\|", text, re.M)),
        "spelled_years": dict(sorted(years.items())),
        "latin_runs": dict(suspicious.most_common()),
        "non_cjk_characters": sorted(non_cjk_characters(spoken)),
    }
    if source_path:
        with open(source_path, encoding="utf-8") as fh:
            report.update(diff_against_source(spoken, fh.read()))

    problems = []
    if suspicious:
        problems.append(f"{len(suspicious)} non-whitelisted Latin/Cyrillic run(s)")
    if report["non_cjk_characters"]:
        problems.append(f"unusual characters: {''.join(report['non_cjk_characters'])}")
    if report.get("unmatched_numbers"):
        problems.append(f"{len(report['unmatched_numbers'])} number(s) not found in source")
    report["problems"] = problems
    return report
