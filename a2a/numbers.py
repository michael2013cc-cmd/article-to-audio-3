#!/usr/bin/env python3
"""Chinese numeral engine: scan, parse, render, normalise.

Ported from the JavaScript engine in ``article-to-audio/scripts/keys.mjs``
(v1). That file is the reason v1 could verify Chinese-spoken numbers at
all, so the behaviour is kept byte-for-byte where it matters and every
quirk is carried over with the note that explains it.

Three jobs:

  * ``scan_chinese_runs`` - find the numeral runs in a text;
  * ``parse_chinese_number`` - read one run as a number;
  * ``renderings`` / ``numeric_forms`` - every way that number could be
    written back down (Arabic, Chinese, scaled by 万/亿).

Why the last one matters: the STT pass does not normalise number format in
one direction. Measured on a real 13-minute transcript:

    script "二零二六年"   -> STT "2026年"    (Chinese -> Arabic)
    script "共 18 篇"     -> STT "共十八篇"  (Arabic -> Chinese!)
    script "百分之八十六" -> STT "86"        (Chinese -> Arabic)

So every numeric key has to accept both forms, and a key picked from the
script's own spelling is not a safe assumption.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------

CN_DIGIT = {
    "零": 0, "〇": 0, "一": 1, "壹": 1, "幺": 1,
    "二": 2, "两": 2, "貳": 2, "贰": 2,
    "三": 3, "參": 3, "叁": 3,
    "四": 4, "肆": 4,
    "五": 5, "伍": 5,
    "六": 6, "陸": 6, "陆": 6,
    "七": 7, "柒": 7,
    "八": 8, "捌": 8,
    "九": 9, "玖": 9,
}

CN_SMALL_UNIT = {"十": 10, "拾": 10, "百": 100, "佰": 100, "千": 1000, "仟": 1000}
CN_BIG_UNIT = {"万": 1e4, "萬": 1e4, "亿": 1e8, "億": 1e8}
CN_POINT = "点點"

# The two scripts of the big units, and what each one is worth.
#
# The simplified form matters more than it looks: a script that says
# 一百四十七万户 is transcribed back as "147万", and a key set that only
# offered "147萬" would call that a miss. v1 generated the traditional
# form only, and it also indexed the unit table with the wrong key when
# the script itself used the traditional character.
CN_BIG_PLAIN = {"万": "万", "萬": "万", "亿": "亿", "億": "亿"}
CN_BIG_TRAD = {"万": "萬", "萬": "萬", "亿": "億", "億": "億"}
CN_BIG_VALUE = {"万": 1e4, "萬": 1e4, "亿": 1e8, "億": 1e8}

# Every character that can begin or continue a Chinese numeral run.
#
# The point characters are NOT in here: 点 is the decimal separator, and
# the scanner has to reach its own branch for it (a run "一点五" is a
# decimal, "一五" is not). Keeping them out of this set is what makes the
# difference between 1.5 and 15.
CN_NUM_CHARS = set(CN_DIGIT) | set(CN_SMALL_UNIT) | set(CN_BIG_UNIT)

# "第三" is a label, not a figure.
CN_ORDINAL = "第"
# "数百万" / "几百万" / "每百万": a unit, not a figure to verify.
CN_VAGUE = "数几每"

CN_DIGITS_STR = "零一二三四五六七八九"
CN_UNITS = ["", "十", "百", "千"]
CN_BIGS = ["", "万", "亿", "万亿"]


def _is_digit(ch: str) -> bool:
    """ASCII digit only.

    ``str.isdigit()`` is True for fullwidth digits and other Unicode
    numerics, which would quietly change how a run is parsed.
    """
    return "0" <= ch <= "9"


def _round_js(x: float) -> float:
    """Round like JavaScript's ``Math.round`` (half away from zero)."""
    if x >= 0:
        return math.floor(x + 0.5)
    return math.ceil(x - 0.5)


def _num_str(value: float) -> str:
    """Format a number the way JavaScript's ``String(n)`` would."""
    if float(value).is_integer() and abs(value) < 1e21:
        return str(int(value))
    return repr(float(value))


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def _read_number_at(s: str, i: int) -> tuple[float, int]:
    """Read a whole number token starting at ``i``: "1,850", "1.5", "13"."""
    j = i
    while j < len(s) and (_is_digit(s[j]) or s[j] == ","):
        j += 1
    raw = s[i:j]
    while raw.endswith(","):
        raw = raw[:-1]
        j -= 1
    value = float(raw.replace(",", "")) if raw else 0.0
    end = j
    if end < len(s) and s[end] == "." and end + 1 < len(s) and _is_digit(s[end + 1]):
        k = end + 1
        while k < len(s) and _is_digit(s[k]):
            k += 1
        value = float(raw.replace(",", "") + s[end:k])
        end = k
    return value, end


def parse_cn_int(s: str) -> float | None:
    """Parse a Chinese integer part (no decimal point). ``None`` if malformed.

    Accepts Arabic digits too, so mixed forms like "3万8千" parse.
    """
    if not s:
        return None
    total = 0.0
    section = 0.0
    num = 0.0
    saw_any = False
    i = 0
    while i < len(s):
        ch = s[i]
        # Spaces and thousands separators are ignorable inside a number run.
        if ch in (" ", "\t", "　", ","):
            i += 1
            continue
        if _is_digit(ch):
            value, end = _read_number_at(s, i)
            num = value
            saw_any = True
            i = end
            continue
        if ch in CN_DIGIT:
            num = CN_DIGIT[ch]
            saw_any = True
            i += 1
            continue
        if ch in CN_SMALL_UNIT:
            section += (num or 1) * CN_SMALL_UNIT[ch]
            num = 0.0
            saw_any = True
            i += 1
            continue
        if ch in CN_BIG_UNIT:
            acc = total + section + num
            total = (acc or 1) * CN_BIG_UNIT[ch]
            section = 0.0
            num = 0.0
            saw_any = True
            i += 1
            continue
        return None
    if not saw_any:
        return None
    # Round: 5.1 x 10^8 is 509999999.99999994 in binary floating point.
    # BUT the rounding must not be applied to large magnitudes: out x 1e6
    # overflows 2^53 (9.007e15) well before 1e8, and the multiply itself
    # loses the precision the rounding was meant to restore. Measured:
    # "294.4 亿" -> 29439999999.999996 instead of 29440000000.
    out = total + section + num
    if not math.isfinite(out):
        return out
    if abs(out) >= 1e6:
        return _round_js(out)
    return _round_js(out * 1e6) / 1e6


def _number_start_before(chars: list[str], idx: int) -> int:
    """Index where the number before ``idx`` starts, or -1."""
    i = idx - 1
    while i >= 0 and chars[i] == " ":
        i -= 1
    if i < 0 or not _is_digit(chars[i]):
        return -1
    j = i
    while j >= 0 and _is_digit(chars[j]):
        j -= 1
    start = j + 1
    # The decimal point may sit BEFORE the integer digits: "1.5 万亿".
    if start - 1 >= 0 and chars[start - 1] == "." and start - 2 >= 0 and _is_digit(chars[start - 2]):
        k = start - 2
        while k >= 0 and _is_digit(chars[k]):
            k -= 1
        start = k + 1
    # A thousands separator may sit before the digits: "1,850 亿" is 1850亿.
    if start - 1 >= 0 and chars[start - 1] == "," and start - 2 >= 0 and _is_digit(chars[start - 2]):
        k = start - 2
        while k >= 0 and (_is_digit(chars[k]) or chars[k] == ","):
            k -= 1
        start = k + 1
    return start


def parse_chinese_number(run: str) -> tuple[float, str | None] | None:
    """Parse a Chinese numeral run into ``(value, big_unit)``.

    Handles: 二十九 / 一百二十八点七 / 三千亿 / 百分之八十六 / 一点五 / 六百.

    ``big_unit`` is "亿"/"万" when the run ended on a large unit, which is
    the case where the STT tends to reformat the number.
    """
    if not run:
        return None
    s = run
    big_unit = None
    if s[-1] in CN_BIG_UNIT:
        big_unit = s[-1]
        s = s[:-1]

    pi = -1
    for idx, c in enumerate(s):
        if c in CN_POINT:
            pi = idx
            break
    if pi >= 0:
        int_part, frac_part = s[:pi], s[pi + 1:]
    else:
        int_part, frac_part = s, ""

    if int_part.startswith("百分之"):
        int_part = int_part[3:]
    if not int_part and not frac_part:
        return None

    if frac_part:
        iv = parse_cn_int(int_part)
        iv = 0.0 if iv is None else iv
    else:
        iv = parse_cn_int(int_part)
        if iv is None:
            return None

    if not frac_part:
        v = iv * CN_BIG_UNIT[big_unit] if big_unit else iv
        if not math.isfinite(v):
            return (v, big_unit)
        return (_round_js(v), big_unit)

    frac = ""
    for ch in frac_part:
        if ch in CN_DIGIT:
            frac += str(CN_DIGIT[ch])
        elif ch == "半":
            frac += "5"
        else:
            return None
    if not frac:
        return None
    dv = iv + float("0." + frac)
    return (_round_js(dv * 1e6) / 1e6, big_unit)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def renderings(value: float, big_unit: str | None = None) -> list[str]:
    """All accepted renderings of a value, most likely first."""
    out: list[str] = []

    def add(x: str) -> None:
        if x not in out:
            out.append(x)

    if float(value).is_integer():
        add(_num_str(value))
        if big_unit:
            plain = CN_BIG_PLAIN[big_unit]
            trad = CN_BIG_TRAD[big_unit]
            base = CN_BIG_VALUE[big_unit]
            scaled = value / base
            if float(scaled).is_integer() and scaled > 0:
                # Both scripts of the unit. The simplified one is what the
                # transcript actually contains.
                add(f"{int(scaled)}{plain}")
                if trad != plain:
                    add(f"{int(scaled)}{trad}")
                alt_scaled = value / 1e4
                if float(alt_scaled).is_integer() and alt_scaled > 0:
                    add(f"{int(alt_scaled)}万")
                    if trad != plain:
                        add(f"{int(alt_scaled)}萬")
    else:
        add(_num_str(value))
        # A decimal with a big unit also has a scaled rendering: "294.4 亿" is
        # 29,440,000,000 once the TTS reads the unit out.
        if big_unit:
            base = CN_BIG_VALUE[big_unit]
            scaled = value * base
            if math.isfinite(scaled) and abs(scaled) >= 1e6:
                add(str(int(_round_js(scaled))))
    return out


def cn_int(n: int) -> str:
    """Integer part in Chinese numerals. 20 -> 二十, 110 -> 一百一十."""
    n = int(n)
    if n == 0:
        return "零"
    if n < 10:
        return CN_DIGITS_STR[n]
    for i in range(len(CN_BIGS) - 1, 0, -1):
        base = 10000 ** i
        if n >= base:
            high = n // base
            rest = n % base
            s = cn_int(high) + CN_BIGS[i]
            if rest:
                s += ("零" if rest < base // 10 else "") + cn_int(rest)
            return s
    d = str(n)
    s = ""
    zero = False
    for i, ch in enumerate(d):
        digit = int(ch)
        unit = CN_UNITS[len(d) - 1 - i]
        if digit == 0:
            zero = True
            continue
        if zero and s:
            s += "零"
        s += CN_DIGITS_STR[digit] + unit
        zero = False
    # 十八, not 一十八
    if 10 <= n < 20 and s.startswith("一"):
        s = s[1:]
    return s


def to_chinese_number(value: float) -> str | None:
    """The Chinese-numeral rendering of an Arabic value."""
    if not math.isfinite(value):
        return None
    neg = value < 0
    v = abs(value)
    int_part = math.floor(v)
    frac = v - int_part
    s = cn_int(int_part)
    if frac > 1e-9:
        digits = (f"{v:.6f}".split(".")[1] or "").rstrip("0")
        s += "点" + "".join(CN_DIGITS_STR[int(c)] for c in digits)
    return ("负" if neg else "") + s


def to_traditional(s: str) -> str:
    """Traditional-Chinese variant of a Chinese numeral string."""
    return s.replace("点", "點").replace("万", "萬").replace("亿", "億")


def numeric_forms(value: float) -> list[str]:
    """Every rendering the STT might plausibly produce for a numeric value.

    Used for tier-A keys too - the script's Arabic form is NOT a guarantee
    that the transcript will be Arabic.
    """
    out: list[str] = []

    def add(x: str) -> None:
        if x not in out:
            out.append(x)

    add(_num_str(value))
    cn = to_chinese_number(value)
    if cn:
        add(cn)
        tw = to_traditional(cn)
        if tw != cn:
            add(tw)
        # A year is spoken digit by digit: 2026 -> 二零二六. The normal
        # rendering (二千零二十六) is correct Chinese but nobody says it,
        # and the transcript is full of years.
        if float(value).is_integer():
            n = int(value)
            if 1900 <= n <= 2099:
                add("".join(CN_DIGITS_STR[int(d)] for d in str(n)))
    # The same number in the other scale, e.g. 1000000000 / 10亿 / 100000万
    if float(value).is_integer():
        for div, unit in ((1e8, "億"), (1e4, "萬")):
            if value % div == 0 and value / div > 0:
                add(f"{_num_str(value / div)}{unit}")
                add(f"{_num_str(value / div)}{'亿' if unit == '億' else '万'}")
    return out


# --------------------------------------------------------------------------
# scanning
# --------------------------------------------------------------------------

def scan_chinese_runs(chars: list[str]) -> list[dict]:
    """THE shared Chinese-numeral scanner.

    Returns one record per run: start, end, run, value, big_unit, has_point,
    anchored.
    """
    runs: list[dict] = []
    i = 0
    n = len(chars)
    while i < n:
        c = chars[i]
        if c not in CN_NUM_CHARS and c not in CN_POINT:
            i += 1
            continue
        # A run must not begin mid-run. `prev` is '' at index 0 and
        # `'' in anything` is True, so the emptiness check must be explicit.
        prev = chars[i - 1] if i > 0 else ""
        if prev and (prev in CN_NUM_CHARS or prev in CN_POINT):
            i += 1
            continue
        # Ordinals are labels, not figures: 第三 / 第一
        if prev == CN_ORDINAL:
            i += 1
            continue
        # 数百万 / 几百万 - a vague quantity, not a figure to verify.
        if prev and prev in CN_VAGUE:
            i += 1
            continue

        # A run may start with an Arabic number when a big unit follows it:
        # "13万4000" is 134000, and the leading 13 belongs to the run.
        start = i
        if c in CN_BIG_UNIT:
            ns = _number_start_before(chars, i)
            if ns >= 0:
                start = ns

        j = i
        saw_point = False
        has_digit = False
        # "百分之" is part of the numeral, not a separate word: 百分之八十六 = 86.
        if c == "百" and j + 2 < n and chars[j + 1] == "分" and chars[j + 2] == "之":
            j += 3
            has_digit = True
        while j < n:
            ch = chars[j]
            # 「千瓦」/「千瓦时」: the 千 belongs to the UNIT (kilowatt), it is
            # not a thousands multiplier. "一百六十七千瓦" is 167 kW -
            # absorbing the 千 produced "7160瓦" (wrong by a factor of ~10).
            if ch in ("千", "仟") and j + 1 < n and chars[j + 1] == "瓦":
                break
            if ch in CN_NUM_CHARS or _is_digit(ch):
                has_digit = True
                j += 1
                continue
            if ch in CN_POINT:
                if saw_point:
                    break
                saw_point = True
                j += 1
                continue
            break
        if not has_digit:
            i += 1
            continue
        run = "".join(chars[start:j])
        # A lone unit character (百 / 千 / 万 / 亿) is a fragment, not a figure.
        if len(run) == 1 and run[0] not in CN_DIGIT:
            i += 1
            continue
        parsed = parse_chinese_number(run)
        if not parsed or not math.isfinite(parsed[0]):
            i += 1
            continue
        value, big_unit = parsed
        # A year is read digit by digit: 二零二六 is 2026, not 26.
        #
        # Read as a plain numeral the run collapses - each digit replaces
        # the last one, so 二零二六 parses as 6 - and a value under ten is
        # then dropped as "a word, not a figure". The v1 implementation had
        # exactly that hole, and the consequence was that a script saying
        # 二零二六年 never matched the key 2026, and the pre-flight check
        # called that key bogus.
        if (
            not big_unit
            and not saw_point
            and len(run) in (3, 4, 5)
            and all(ch in CN_DIGIT for ch in run)
            and j < n
            and chars[j] == "年"
        ):
            value = float(int("".join(str(CN_DIGIT[ch]) for ch in run)))
        # Bare small integers are words, not figures ("一个", "两次").
        if not big_unit and not saw_point and value < 10:
            i += 1
            continue
        # A run anchored by an Arabic prefix ("5.1 亿") already has a tier-A
        # candidate for the Arabic part, which is the robust key. The derived
        # value (510000000) is a form the STT is unlikely to produce.
        anchored = start < i and _is_digit(run[0])
        runs.append({
            "start": start,
            "end": j,
            "run": run,
            "value": value,
            "big_unit": big_unit,
            "has_point": saw_point,
            "anchored": anchored,
        })
        i = j
    return runs


def normalize_script(text: str) -> str:
    """Text with every Chinese numeral run replaced by its Arabic rendering."""
    chars = list(text)
    runs = scan_chinese_runs(chars)
    by_start = {r["start"]: r for r in runs}
    out: list[str] = []
    i = 0
    while i < len(chars):
        r = by_start.get(i)
        if r:
            out.append(renderings(r["value"], r["big_unit"])[0])
            i = r["end"]
        else:
            out.append(chars[i])
            i += 1
    return "".join(out)


def context(chars: list[str], start: int, end: int, width: int = 26) -> str:
    """A readable window of text around a match."""
    s = "".join(chars[max(0, start - width):end + width])
    return s.replace("\n", " ").strip()
