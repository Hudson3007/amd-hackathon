"""Thin client for Mini-Challenge 3.

This file is a new process for every index and every query, so it must not load
a model or read the corpus. It forwards work to the long-running daemon over a
unix socket and does nothing else.

    python3 /app/app.py --index /app/corpus
    python3 /app/app.py --corpus /app/corpus --query-id query_01 --query "..."

The contract is that a JSON file always lands in /app/output, whatever happens.
A missing or malformed file scores zero, so the write is wrapped end to end and
falls back to an empty answer rather than raising.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wire  # noqa: E402

OUTPUT_DIR = os.environ.get("MC3_OUTPUT", "/app/output")
CONNECT_TIMEOUT = float(os.environ.get("MC3_CONNECT_TIMEOUT", "25"))


def request(payload, timeout):
    sock = wire.client_connect(time.time() + CONNECT_TIMEOUT)
    try:
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        buf = bytearray()
        sock.settimeout(timeout)
        while b"\n" not in buf:
            chunk = sock.recv(65536)
            if not chunk:
                raise OSError("daemon closed the connection")
            buf.extend(chunk)
        return json.loads(bytes(buf).split(b"\n", 1)[0].decode("utf-8"))
    finally:
        try:
            sock.close()
        except OSError:
            pass


def write_output(query_id, result):
    """Write the required JSON, atomically, with every field always present."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    target = os.path.join(OUTPUT_DIR, f"{query_id}_output.json")

    answer = result.get("answer", "")
    if not isinstance(answer, str):
        answer = "" if answer is None else str(answer)

    citations = result.get("citations", [])
    if not isinstance(citations, list):
        citations = []
    citations = [c for c in citations if isinstance(c, str)]

    confidence = result.get("confidence", 0.0)
    try:
        confidence = float(confidence)
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = min(1.0, max(0.0, confidence))

    payload = {"answer": answer, "citations": citations, "confidence": confidence}

    tmp = f"{target}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    os.replace(tmp, target)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", metavar="DIR")
    parser.add_argument("--corpus", metavar="DIR")
    parser.add_argument("--query-id", dest="query_id")
    parser.add_argument("--query")
    args, _unknown = parser.parse_known_args()

    if args.index:
        # Charged to the startup budget, so it is allowed to be slow.
        try:
            request({"op": "index", "root": args.index}, timeout=3600)
        except Exception as exc:
            print(f"[mc3] index failed: {exc}", file=sys.stderr)
            return 0
        return 0

    if args.corpus and args.query is not None:
        query_id = args.query_id
        if not query_id:
            print("[mc3] missing --query-id", file=sys.stderr)
            return 0
        try:
            result = request(
                {"op": "query", "root": args.corpus, "query": args.query},
                timeout=600,
            )
        except Exception as exc:
            print(f"[mc3] query failed: {exc}", file=sys.stderr)
            result = {"answer": "", "citations": [], "confidence": 0.0}
        try:
            write_output(query_id, result)
        except Exception as exc:
            print(f"[mc3] could not write output: {exc}", file=sys.stderr)
        return 0

    parser.print_usage(file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
