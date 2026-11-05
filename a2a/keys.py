#!/usr/bin/env python3
"""Verification-key extraction, ranking, and pre-flight validation.

Ported from ``article-to-audio/scripts/keys.mjs`` (v1).

WHY THIS EXISTS
Hand-written verify lists drift from the script. A key that was never in
the narration can never appear in the transcript, so it reports as a "miss"
that costs a full TTS+STT cycle to chase. That happened three times in a
row during the Episode 3 build (104 / 158 / 69 - none of them were in the
script).

The fix is structural, not behavioural: keys are DERIVED from the script,
and any hand-written key that is absent from the script is rejected before
a single API call is made.

Two tiers:
  A - literal Arabic numerals in the script ("59.2%", "0.993", "3.1").
      STT echoes these almost verbatim. Safest anchors. Prefer these.
  B - Chinese numerals in the script ("百分之八十六", "六百美元", "三点一"),
      converted to Arabic here. The STT converts them too, but the
      conversion is not guaranteed (units get reformatted, decimals drift),
      so each tier-B key carries a set of acceptable renderings and the
      verifier accepts ANY.

A key is therefore a set of alternatives, not a single string.
"""

from __future__ import annotations

import re

from .numbers import (
    CN_BIG_UNIT,
    CN_DIGIT,
    CN_NUM_CHARS,
    CN_POINT,
    _is_digit,
    context,
    numeric_forms,
    parse_chinese_number,
    renderings,
    scan_chinese_runs,
)

# Proper nouns that contain numerals - a name, not a figure.
NAME_ALLOW = ["零一万物", "零零后", "一九八四"]


def _read_number_token(chars: list[str], i: int) -> tuple[str, int, str]:
    """Read a tier-A numeral starting at ``i``.

    Returns ``(text, end_index, suffix)``.
    """
    n = len(chars)
    j = i
    while j < n and _is_digit(chars[j]):
        j += 1
    # Thousands separators: "1,048,576" is one number, not three.
    while (
        j < n and chars[j] == "," and j + 3 < n
        and _is_digit(chars[j + 1]) and _is_digit(chars[j + 2]) and _is_digit(chars[j + 3])
    ):
        j += 4
        while j < n and _is_digit(chars[j]):
            j += 1
    # A decimal part may follow the comma-grouped integer: "7,065.6" is ONE
    # number. Without this the scanner stopped at "7,065" and dropped ".6".
    if j < n and chars[j] == "." and j + 1 < n and _is_digit(chars[j + 1]):
        j += 1
        while j < n and _is_digit(chars[j]):
            j += 1
    k = j
    while k < n and chars[k] in "％%万萬亿億倍x×$￥":
        k += 1
    return "".join(chars[i:j]), j, "".join(chars[j:k])


def extract_candidates(segments: list[str]) -> list[dict]:
    """Scan text for numeric candidates.

    Returns records with: key, alts, tier, value, segment, context, count.
    tier 'A' = literal Arabic in the script, tier 'B' = converted from
    Chinese.
    """
    seen: dict[str, dict] = {}

    def push(c: dict) -> None:
        k = c["key"]
        if k in seen:
            prev = seen[k]
            prev["count"] += 1
            prev["alts"] = list(dict.fromkeys([*prev["alts"], *c["alts"]]))
            # A literal-Arabic sighting always wins the tier label.
            if c["tier"] == "A":
                prev["tier"] = "A"
            return
        seen[k] = c

    for si, text in enumerate(segments):
        chars = list(text)

        # ---- Tier A: literal Arabic numerals -----------------------------
        i = 0
        while i < len(chars):
            if not _is_digit(chars[i]):
                i += 1
                continue
            # A digit followed by a Chinese unit belongs to a Chinese-number
            # run ("3万8千"), which the Chinese scanner handles as one value.
            if i + 1 < len(chars) and chars[i + 1] in CN_NUM_CHARS:
                i += 1
                continue
            num, j, suffix = _read_number_token(chars, i)
            if not num or num == ".":
                i = max(j, i + 1)
                continue
            try:
                value = float(num.replace(",", ""))
            except ValueError:
                i = max(j, i + 1)
                continue
            if value != value or value in (float("inf"), float("-inf")):
                i = max(j, i + 1)
                continue

            # A trailing big unit scales the value: the script's "294.4 亿" is
            # read out as 二百九十四点四亿, which normalises to "29440000000" -
            # a rendering the key "294.4" would never match. The house style
            # puts a space between the figure and a Chinese unit, so the unit
            # is also looked for one space further out.
            alt_list = [num, *numeric_forms(value), num.replace(",", "")]
            win = "".join(chars[j:j + 4]).lstrip(" ")
            sfx = suffix.lstrip(" ")
            # 万亿 / 万亿 is a COMPOUND unit (万 x 亿 = 1e12), not 万 (1e4).
            big_c = None
            if sfx and sfx[0] in "万萬亿億":
                big_c = sfx[0]
            elif win and win[0] in "万萬亿億":
                big_c = win[0]
            if big_c and not float(value).is_integer():
                is_yi = big_c in "亿億"
                is_wan = big_c in "万萬"
                base = 1e12 if (is_yi and is_wan) else (1e8 if is_yi else 1e4)
                scaled = value * base
                if scaled == scaled and scaled not in (float("inf"), float("-inf")):
                    alt_list.append(str(int(round(scaled))))
            push({
                "key": num,
                # The literal form belongs in the alt set: "46,225" must match a
                # transcript that says "46,225", not only the comma-free variant.
                "alts": list(dict.fromkeys(alt_list)),
                "tier": "A",
                "value": value,
                "segment": si,
                "count": 1,
                "context": context(chars, i, j + len(suffix)),
                "has_unit": bool(suffix),
            })
            i = max(j + len(suffix), i + 1)

        # ---- Tier B: Chinese numerals -------------------------------------
        for r in scan_chinese_runs(chars):
            if r["anchored"]:
                continue  # tier A already covers the Arabic part
            alts = renderings(r["value"], r["big_unit"])
            push({
                "key": alts[0],
                "alts": alts,
                "tier": "B",
                "value": r["value"],
                "segment": si,
                "count": 1,
                "context": context(chars, r["start"], r["end"]),
                "has_unit": bool(r["big_unit"]),
                "cn_text": r["run"],
            })

    return list(seen.values())


def rank_candidates(candidates: list[dict], top_n: int = 25) -> list[dict]:
    """Rank candidates for verification value, then spread the picks across
    segments so a single dense paragraph cannot monopolise the key budget."""
    scored = []
    for c in candidates:
        s = 0
        # NOTE: tier A no longer means "more reliable" - the STT renders
        # numbers in either form regardless of how the script wrote them.
        # What still matters is that the figure is distinctive and carries a
        # unit.
        if c.get("has_unit"):
            s += 4                     # 59.2% / 3.1x / 300万
        if "." in str(c["value"]):
            s += 3                     # 0.993 / 128.7 - precise figures
        if c["value"] >= 100:
            s += 2                     # distinctive even in Chinese form
        if c["value"] >= 10:
            s += 1
        if c["count"] == 1:
            s += 1                     # distinctive
        if c["value"] < 10:
            s -= 2                     # 3 / 5 - too generic to prove anything
        if c["value"] < 10 and c["count"] > 3:
            s -= 3
        if c["tier"] == "B" and c.get("has_unit"):
            s -= 2                     # unit reformatting risk
        scored.append({**c, "score": s})

    scored.sort(key=lambda c: (-c["score"], -c["value"]))

    per_seg: dict[int, int] = {}
    picked: list[dict] = []
    rest: list[dict] = []
    for c in scored:
        n = per_seg.get(c["segment"], 0)
        if n < 2:
            picked.append(c)
            per_seg[c["segment"]] = n + 1
        else:
            rest.append(c)
    for c in rest:
        if len(picked) >= top_n:
            break
        picked.append(c)
    return picked[:top_n]


def _token_internal(before: str, before2: str, after: str) -> bool:
    """True when a number sitting between these neighbours belongs to a
    larger Latin token - an identifier, not a figure.

    The three shapes, all measured on real scripts (v1, preflight-all.mjs):

      1. the char before is a letter            -> MI355X / gfx950 / HBM4
      2. the char before is - or . and the one
         before THAT is alnum                   -> GLM-5.2 / Llama-3.3
      3. the char before is - and the one after
         is a letter                            -> the 70 in Llama-3.3-70B

    A digit preceded by a digit does not count: the 5 in 13.5 is part of a
    real figure. And only the PRECEDING side is judged (v1's first version
    required both sides to be tokenish and let "GLM-5.2、MiniMax" through
    because a full-width 、 follows the number - 2026-10-23 fix).
    """
    if re.match(r"[A-Za-z]", before):
        return True
    if re.match(r"[-.]", before) and re.match(r"[A-Za-z0-9]", before2):
        return True
    if before == "-" and re.match(r"[A-Za-z]", after):
        return True
    return False


def suspect_model_keys(keys: list, segments: list[str]) -> list[str]:
    """Keys whose EVERY occurrence in the script sits inside a Latin token.

    MI355X's 355, gfx950's 950, H100's 100, Llama-3.3-70B's 70: those digits
    really are in the script, but they are model names, not data. The TTS
    mangles them as often as not, so a listen-back reports a miss that no
    re-render will ever fix - a guaranteed false alarm dressed up as a real
    one.

    The gate is deliberately "every occurrence": one free-standing hit means
    the number also exists as a figure somewhere, and a real figure next to
    an English word is common enough that this stays a WARNING, never a
    silent drop (hand the list to a human, or let --auto-keys skip them
    with a printed note).
    """
    whole = "\n".join(segments)
    out: list[str] = []
    for k in keys:
        label = str(k.get("key", "")) if isinstance(k, dict) else str(k)
        n = label.replace(",", "")
        if not n or not re.search(r"\d", n):
            continue
        hits: list[bool] = []
        for m in re.finditer(re.escape(n), whole):
            start = m.start()
            before = whole[start - 1] if start >= 1 else ""
            before2 = whole[start - 2] if start >= 2 else ""
            end = m.end()
            after = whole[end] if end < len(whole) else ""
            hits.append(_token_internal(before, before2, after))
            if len(hits) >= 6:  # enough evidence either way
                break
        if hits and all(hits):
            out.append(label)
    return out


def preflight(keys: list, segments: list[str]) -> list[str]:
    """Which of these keys are not actually in the script?

    A key counts as present if it appears in the raw script OR in the
    normalised script OR as any of its accepted renderings.
    """
    from .numbers import normalize_script

    raw = "\n".join(segments)
    norm = normalize_script(raw)
    bogus: list[str] = []
    for k in keys:
        if isinstance(k, dict):
            label = str(k.get("key"))
            forms = k.get("alts") if isinstance(k.get("alts"), list) and k.get("alts") else [label]
        else:
            label = str(k)
            forms = [label]
        found = any(f in raw or f in norm for f in forms) or label in raw or label in norm
        if not found:
            bogus.append(label)
    return bogus


def lint_numbers(segments: list[str]) -> list[dict]:
    """Rendering warnings - editorial, not blocking.

    Big numbers written in Chinese are hard for the STT to reproduce
    faithfully, and decimals written in Chinese often lose their point.
    Both are worth writing as Arabic in the source script.
    """
    warns: list[dict] = []
    for si, text in enumerate(segments):
        chars = list(text)
        for r in scan_chinese_runs(chars):
            # A numeral inside a proper noun is part of the name, not a figure.
            if any(n in r["run"] for n in NAME_ALLOW):
                continue
            # Anchored by an Arabic prefix ("5.1 亿"): the Arabic part is the
            # key, and the unit is unambiguous enough for the STT to get right.
            if r["anchored"]:
                continue
            big = r["value"] >= 100000 or bool(r["big_unit"])
            if not big and not r["has_point"]:
                continue
            # "三百万亿" is ambiguous: 3x10^11 (三千亿) or 3x10^14 (literal
            # parse). Both humans and STT get it wrong, so no value is
            # suggested - rewrite it.
            has_wan = any(w in r["run"] for w in "万萬")
            has_yi = any(y in r["run"] for y in "亿億")
            ambiguous = has_wan and has_yi
            warns.append({
                "segment": si + 1,
                "run": r["run"],
                "value": r["value"],
                "ambiguous": ambiguous,
                "suggest": None if ambiguous else str(int(r["value"])) if float(r["value"]).is_integer() else str(r["value"]),
                "reason": (
                    "量级歧义（同时含「万」与「亿」）—— 建议改成「三千亿」这类无歧义写法，或直接写阿拉伯数字"
                    if ambiguous
                    else "中文小数 —— STT 容易丢小数点，建议写阿拉伯数字"
                    if r["has_point"]
                    else "中文大数 —— STT 容易改写量级，建议写阿拉伯数字"
                ),
                "context": context(chars, r["start"], r["end"], 18),
            })
    return warns
