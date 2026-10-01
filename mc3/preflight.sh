#!/usr/bin/env bash
# Self-check for the Mini-Challenge 3 image.
#
#   docker run -d --name mc3 --cap-drop DAC_OVERRIDE mc3-rag:v1 sleep infinity
#   docker exec mc3 /app/preflight.sh
#
# These are the same checks the grading pipeline runs. Run it on Linux. The
# chmod 000 case cannot be reproduced on Windows at all, and running as root
# does not stop it either, because root bypasses mode bits - hence
# --cap-drop DAC_OVERRIDE above, which is what the graders drop.

set -uo pipefail

CORPUS="${CORPUS:-/app/corpus}"
OUT="${OUT:-/app/output}"
PASS=0
FAIL=0

ok()   { echo "  PASS  $1"; PASS=$((PASS+1)); }
bad()  { echo "  FAIL  $1${2:+  -> $2}"; FAIL=$((FAIL+1)); }

echo "[1] ROCm torch survived every pip install"
VER=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
case "$VER" in
  *+rocm*) ok "torch is a ROCm build ($VER)" ;;
  *)       bad "torch is not a ROCm build" "got '${VER:-<none>}'; a CUDA wheel was installed over the base" ;;
esac

echo
echo "[2] startup, model load and the --index pass fit the 10 minute budget"
START=$(date +%s)
if python3 /app/app.py --index "$CORPUS"; then
  ELAPSED=$(( $(date +%s) - START ))
  if [ "$ELAPSED" -le 600 ]; then
    ok "--index completed in ${ELAPSED}s (budget 600s)"
  else
    bad "--index took ${ELAPSED}s" "over the 600s startup budget"
  fi
else
  bad "--index exited non-zero"
fi

echo
echo "[3] every --query writes a well-formed file inside 30s"
run_query() {
  local qid="$1" question="$2"
  local t0 t1 out="$OUT/${qid}_output.json"
  rm -f "$out"
  t0=$(date +%s)
  python3 /app/app.py --corpus "$CORPUS" --query-id "$qid" --query "$question" >/dev/null 2>&1
  t1=$(date +%s)
  if [ ! -f "$out" ]; then bad "${qid} wrote no file" "file missing"; return; fi
  if [ $((t1-t0)) -gt 30 ]; then bad "${qid} took $((t1-t0))s" "over the 30s budget"; return; fi
  if python3 - "$out" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert isinstance(d.get("answer"), str), "answer must be a string"
assert isinstance(d.get("citations"), list), "citations must be a list"
assert all(isinstance(c, str) for c in d["citations"]), "citations must be strings"
assert isinstance(d.get("confidence"), (int, float)), "confidence must be a number"
PY
  then ok "${qid} valid in $((t1-t0))s"
  else bad "${qid} is malformed" "missing or wrong-typed field"
  fi
}

run_query query_01 "What is the maximum junction temperature of the TQ-40?"
run_query query_02 "Which firmware version fixed ticket ORR-1847?"
run_query query_04 "What is the unit price of the TQ-40 at 10,000 unit volume?"

echo
echo "[4] the unanswerable question returns an empty answer and no citations"
if [ -f "$OUT/query_04_output.json" ]; then
  if python3 - "$OUT/query_04_output.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
assert d["answer"] == "", f"answer should be empty, got {d['answer']!r}"
assert d["citations"] == [], f"citations should be empty, got {d['citations']!r}"
PY
  then ok "unanswerable question refused correctly"
  else bad "unanswerable question did not refuse" "guessing here is scored wrong"
  fi
fi

echo
echo "[5] container is still running after repeated exec"
if kill -0 1 2>/dev/null || pgrep -f daemon.py >/dev/null 2>&1; then
  ok "daemon still alive"
else
  bad "daemon is gone" "the harness execs repeatedly; it must stay up"
fi

echo
echo "$PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
