"""Checks finalize() against the ten sample expectations in the challenge PDF.

Run: python tests/test_normalize.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from normalize import compare_key, finalize, matches  # noqa: E402

# (label, plausible VLM output, expected answer)
CASES = [
    (
        "1 US plate California, clean",
        "7ABC123",
        "7ABC123",
    ),
    (
        "1 US plate California, banner leaked into output",
        "CALIFORNIA 7ABC123",
        "7ABC123",
    ),
    (
        "1 US plate California, slogan leaked too",
        "CALIFORNIA 7ABC123 THE GOLDEN STATE",
        "7ABC123",
    ),
    (
        "2 Chinese plate Beijing",
        "\u4eacA\u00b712345",
        "\u4eacA\u00b712345",
    ),
    (
        "2 Chinese plate Beijing, banner leaked",
        "BEIJING \u4eacA\u00b712345",
        "\u4eacA\u00b712345",
    ),
    (
        "3 US plate New York, angled",
        "JHT 2951",
        "JHT 2951",
    ),
    (
        "3 US plate New York, no space",
        "JHT2951",
        "JHT 2951",
    ),
    (
        "4 US plate motion blur",
        "5XYZ891",
        "5XYZ891",
    ),
    (
        "5 Chinese plate Shanghai, glare",
        "\u6caaB\u00b788888",
        "\u6caaB\u00b788888",
    ),
    (
        "6 Stop sign",
        "STOP",
        "STOP",
    ),
    (
        "7 Stop sign with noise",
        "STOP",
        "STOP",
    ),
    (
        "8 Speed limit sign",
        "SPEED LIMIT 65",
        "SPEED LIMIT 65",
    ),
    (
        "8 Speed limit sign, lowercased by model",
        "Speed Limit 65",
        "SPEED LIMIT 65",
    ),
    (
        "9 Work zone sign, three lines",
        "ROAD WORK AHEAD",
        "ROAD WORK AHEAD",
    ),
    (
        "9 Work zone sign, model returned it newline separated",
        "ROAD\nWORK\nAHEAD",
        "ROAD WORK AHEAD",
    ),
    (
        "10 Advisory speed plaque",
        "35",
        "35",
    ),
    (
        "10 Advisory plaque, model added a unit",
        "35 MPH",
        "35",
    ),
    # robustness beyond the published samples
    (
        "Texas plate printed with a space",
        "7ABC 123",
        "7ABC123",
    ),
    (
        "plate with model preamble",
        "Text: 7ABC123",
        "7ABC123",
    ),
    (
        "plate wrapped in quotes and markdown",
        '**"7ABC123"**',
        "7ABC123",
    ),
    (
        "massachusetts style short plate",
        "7AB 123",
        "7AB123",
    ),
    (
        "empty model output",
        "",
        "",
    ),
]

EXPECTED_FAILS = [
    # A wrong character must not be silently repaired.
    ("misread character", "7ABC128", "7ABC123"),
    # We keep every word on a sign, so extra prose genuinely loses.
    ("sign plus model commentary", "STOP. The sign is red and octagonal.", "STOP"),
]


def main():
    failures = []
    for label, raw, expected in CASES:
        got = finalize(raw)
        ok = compare_key(got) == compare_key(expected)
        if not ok:
            failures.append(f"FAIL  {label}\n      got      {got!r}\n      expected {expected!r}")
        else:
            print(f"ok    {label}")

    for label, raw, expected in EXPECTED_FAILS:
        if matches(raw, expected):
            failures.append(f"FAIL  {label}\n      unexpectedly matched {expected!r}")
        else:
            print(f"ok    {label} (correctly rejected)")

    print()
    if failures:
        print("\n".join(failures))
        print(f"\n{len(failures)} failure(s)")
        return 1
    print(f"all {len(CASES) + len(EXPECTED_FAILS)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
