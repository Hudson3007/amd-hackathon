"""Retrieval-recall regression test for MC3.

The other MC3 tests are structural: they prove the daemon starts, the socket
works and the reply parses. None of them prove that retrieval finds the file a
question is actually answered from. A regression there is silent -- the daemon
answers confidently from the wrong document, or returns nothing, and the run
still reports success.

This test indexes the synthetic corpus and asserts the gold file lands in the
retrieved set for each question. Image chunks are stubbed, because captioning
needs the VLM and this test must run on CPU; the caption path is covered by
tests/test_mc3.py.
"""

import importlib.util
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import make_mc3_corpus as maker

DAEMON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mc3", "daemon.py"
)

STUB_DESCRIPTIONS = {
    "specs/backplane_pinout.png": "Backplane pinout. B14 = THERM_ALERT#. B15 = FAN_PWM#.",
}

CASES = [
    ("specs/tq40_datasheet_r2.pdf", "What is the recommended operating temperature range for the TQ40?"),
    ("specs/tq40_datasheet_r2.pdf", "TQ40-2200 supply voltage tolerance"),
    ("planning/roadmap_fy27.docx", "Which quarter is the Falcon bridge milestone targeted for?"),
    ("planning/roadmap_fy27.docx", "What is the roadmap for FY27 phase 2?"),
    ("support/rma_parts.xlsx", "What is the RMA lead time for part 4471-B?"),
    ("support/rma_parts.xlsx", "4471-B replacement cycle"),
    ("specs/backplane_pinout.png", "What signal is on pin B14 of the backplane?"),
]


def load_daemon(corpus):
    os.environ["MC3_NO_MODEL"] = "1"
    os.environ["MC3_CORPUS"] = corpus
    os.environ["MC3_SOCKET"] = os.path.join(tempfile.gettempdir(), "mc3_recall.sock")
    spec = importlib.util.spec_from_file_location("mc3_daemon_recall", DAEMON)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    corpus = os.path.join(tempfile.gettempdir(), "mc3_recall_corpus")
    if not os.path.isdir(corpus):
        maker.OUT = corpus
        maker.build()

    daemon = load_daemon(corpus)
    daemon.describe_image = lambda relpath: STUB_DESCRIPTIONS.get(relpath)
    daemon.build_index(corpus)

    failures = []
    for gold, question in CASES:
        chunks, _ = daemon.retrieve(question)
        found = list(daemon.group_by_file(chunks))
        if gold in found:
            print(f"  PASS rank={found.index(gold) + 1} :: {question[:56]}")
        else:
            failures.append((gold, question, found[:6]))
            print(f"  FAIL :: {question[:56]} -> {found[:6]}")

    recall = (len(CASES) - len(failures)) / len(CASES)
    print(f"\nrecall {len(CASES) - len(failures)}/{len(CASES)} = {recall:.0%}")
    if failures:
        print(f"FAILED: {len(failures)} question(s) did not retrieve the gold file")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
