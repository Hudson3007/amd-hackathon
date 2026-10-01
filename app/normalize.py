"""Transcription rules for the AMD AI Academy Mini-Challenge 2 (OCR).

Grading is pass/fail per image after the harness normalizes both the expected
answer and ours: uppercase, all whitespace removed, and - . middot _ removed.
So the only thing that matters is which characters we emit, not how we format.

Categories are distinguished structurally rather than by a classifier:
  - CJK anywhere in the string        -> license plate, keep province prefix
  - one token mixing letters+digits    -> plate number, drop banners/slogans
  - letters + digits as two tokens    -> plate printed with a space (e.g. JHT 2951)
  - everything else                    -> road sign, keep all text
"""

import re

CJK_RANGES = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
CJK_RE = re.compile(f"[{CJK_RANGES}]")
CJK_PLATE_RE = re.compile(f"[{CJK_RANGES}]{{1,2}}[A-Z]?[^0-9{CJK_RANGES}]*[0-9A-Z]{{4,8}}")

GRADER_STRIP = " \t\n\r-.·_\u2013\u2014/"
_GRADER_RE = re.compile(f"[{re.escape(GRADER_STRIP)}]+")

ALPHA_RE = re.compile(r"^[A-Z]+$")
DIGIT_RE = re.compile(r"^[0-9]+$")
PLATE_RE = re.compile(r"^[A-Z0-9]{4,8}$")

# Advisory plaques carry a bare number, but models like to append the unit.
# Restricting the strip to tokens that follow a number keeps genuine sign text
# such as "SPEED LIMIT 65" intact, because there the words precede the number.
UNIT_RE = re.compile(r"^(MPH|KPH|KM/?H(R)?|M/?S|KG|LBS?|FT|FT/ND)$")

_NOISE_PREFIX_RE = re.compile(
    r"^\s*(?:the\s+)?(?:text|transcription|answer|output|result|"
    r"license\s+plate|licence\s+plate|plate|sign|number)\s*[:\-–]\s*",
    re.IGNORECASE,
)


def clean(raw):
    """Strip chatty scaffolding the VLM may add around the transcription."""
    if raw is None:
        return ""
    s = str(raw)
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"[*_`~#>|\[\]{}()]+", " ", s)
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = _NOISE_PREFIX_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip("\"'`\u201c\u201d\u2018\u2019")


def compare_key(s):
    """Replicate the harness normalization, for local assertions only."""
    return _GRADER_RE.sub("", (s or "").upper())


def _tokens(s):
    return [t for t in re.split(r"[\s,;|]+", s.strip()) if t]


def _is_mixed(t):
    return bool(re.search(r"[A-Z]", t)) and bool(re.search(r"[0-9]", t))


def _looks_like_plate(t):
    return bool(PLATE_RE.match(t))


def _finalize_cjk(s):
    m = CJK_PLATE_RE.search(s)
    return m.group(0) if m else s


def _strip_units(tokens):
    kept = []
    for t in tokens:
        if kept and DIGIT_RE.match(kept[-1]) and ALPHA_RE.match(t) and UNIT_RE.match(t):
            continue
        kept.append(t)
    return kept


def _finalize_ascii(s):
    tokens = _strip_units(_tokens(s))
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]

    plate_idx = [
        i for i, t in enumerate(tokens) if _is_mixed(t) and _looks_like_plate(t)
    ]
    if len(plate_idx) == 1:
        i = plate_idx[0]
        cluster = [tokens[i]]
        for j in (i - 1, i + 1):
            if 0 <= j < len(tokens) and DIGIT_RE.match(tokens[j]):
                cluster.append(tokens[j])
        joined = "".join(cluster)
        if _is_mixed(joined) and _looks_like_plate(joined):
            return joined
        return tokens[i]

    if len(tokens) == 2:
        a, b = tokens
        if (
            ALPHA_RE.match(a)
            and DIGIT_RE.match(b)
            and 2 <= len(a) <= 4
            and 1 <= len(b) <= 4
        ):
            return f"{a} {b}"

    return " ".join(tokens)


def finalize(raw):
    """Raw VLM text -> the string we submit."""
    s = clean(raw)
    if not s:
        return ""
    if CJK_RE.search(s):
        return _finalize_cjk(s.upper())
    return _finalize_ascii(s.upper())


def matches(raw, expected):
    """True when our finalized output would pass the harness comparison."""
    return compare_key(finalize(raw)) == compare_key(expected)
