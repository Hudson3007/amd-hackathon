"""Mini-Challenge 2 entrypoint.

Contract enforced by the grading harness:
    python3 /app/app.py --input-image /app/input/image_01.png
    -> /app/output/image_01_output.json  {"text": "...", "confidence": 0.0-1.0}

Always writes a schema-valid file and exits 0, so one bad image costs 20
points instead of ending the run.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import transcribe  # noqa: E402

DEFAULT_INPUT_DIR = os.environ.get("OCR_INPUT_DIR", "/app/input")
DEFAULT_OUTPUT_DIR = os.environ.get("OCR_OUTPUT_DIR", "/app/output")
TIME_BUDGET = float(os.environ.get("OCR_TIME_BUDGET", "27"))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--input-image", default=None)
    p.add_argument("--input-dir", default=None)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--model", default=None)
    return p.parse_args(argv)


def output_path(image_path, output_dir):
    return Path(output_dir) / f"{Path(image_path).stem}_output.json"


def write_result(path, text, confidence):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "text": text or "",
        "confidence": round(float(confidence), 4),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return payload


def handle(image_path, output_dir, model=None):
    started = time.time()
    text, confidence = "", 0.0
    try:
        text, confidence = transcribe(str(image_path), model_path=model)
    except Exception as exc:
        print(f"[warn] {Path(image_path).name}: {exc}", file=sys.stderr)

    elapsed = time.time() - started
    if not text and elapsed > TIME_BUDGET:
        print(f"[warn] no output within budget for {Path(image_path).name}", file=sys.stderr)

    payload = write_result(output_path(image_path, output_dir), text, confidence)
    print(
        f"[ok] {Path(image_path).name} -> {payload['text']!r} "
        f"conf={payload['confidence']} in {elapsed:.1f}s"
    )
    return payload


def main(argv=None):
    args = parse_args(argv)

    if args.input_image:
        handle(args.input_image, args.output_dir, args.model)
        return 0

    # Local development convenience: the harness never uses this path.
    input_dir = Path(args.input_dir or DEFAULT_INPUT_DIR)
    if not input_dir.is_dir():
        print(f"[error] no --input-image and not a directory: {input_dir}", file=sys.stderr)
        write_result(Path(args.output_dir) / "error_output.json", "", 0.0)
        return 0

    images = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"[warn] no PNG/JPEG/TIFF images in {input_dir}", file=sys.stderr)
    for image in images:
        handle(image, args.output_dir, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
