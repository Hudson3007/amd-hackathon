#!/usr/bin/env bash
# Replicates the five checks AMD runs before it scores anything. Any failure here
# is a zero, so run this before submitting. Linux or WSL2 with Docker.
#
#   IMAGE=myrepo/ocr:tag ./scripts/preflight.sh
#   SAMPLES=./samples IMAGE=myrepo/ocr:tag ./scripts/preflight.sh

set -uo pipefail

IMAGE="${IMAGE:?set IMAGE=<repo>:<tag>}"
SAMPLES="${SAMPLES:-./samples}"
BASE="${BASE:-rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0}"
MAX_BYTES=$((60 * 1024 * 1024 * 1024))
STARTER="rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0"
fails=0

pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; fails=$((fails + 1)); }
note() { printf '  ....  %s\n' "$1"; }

echo "== gate 1: base image layer identity =="
if ! docker image inspect "$BASE" >/dev/null 2>&1; then
  note "base not local, pulling (one time, large)"
  docker pull "$BASE" >/dev/null 2>&1
fi
base_layers=$(docker image inspect --format '{{range .RootFS.Layers}}{{println .}}{{end}}' "$BASE" 2>/dev/null)
img_layers=$(docker image inspect --format '{{range .RootFS.Layers}}{{println .}}{{end}}' "$IMAGE" 2>/dev/null)
n=$(printf '%s\n' "$base_layers" | sed '/^$/d' | wc -l)
if printf '%s\n' "$base_layers" | sed '/^$/d' > /tmp/.base_layers
then
  if head -n "$n" /tmp/.base_layers | diff -q - <(printf '%s\n' "$img_layers" | sed '/^$/d' | head -n "$n") >/dev/null 2>&1; then
    pass "lower $n layers match $BASE"
  else
    fail "lower layers diverge from the base (squashed or wrong base) -> disqualified"
  fi
else
  fail "could not read layer list for $BASE"
fi

echo "== gate 2: uncompressed image size (limit 60 GiB) =="
size=$(docker image inspect --format '{{.Size}}' "$IMAGE" 2>/dev/null || echo 0)
if [ "$size" -gt 0 ] && [ "$size" -le "$MAX_BYTES" ]; then
  pass "$((size / 1024 / 1024)) MiB of 61440 MiB"
else
  fail "image is $((size / 1024 / 1024)) MiB"
fi

echo "== gate 3: contract paths exist =="
for p in /app/app.py /app/requirements.txt; do
  if docker run --rm "$IMAGE" test -f "$p" >/dev/null 2>&1; then pass "$p present"; else fail "$p missing"; fi
done
if docker run --rm "$IMAGE" test -d /models >/dev/null 2>&1; then
  note "/models present (weights bundled, so no runtime download)"
else
  note "/models absent - app must download weights at startup (costs startup budget)"
fi

echo "== gate 4: no secrets baked into the image =="
if docker run --rm "$IMAGE" sh -c 'ls -A /app 2>/dev/null | grep -qi "^\.env$"'; then
  fail "/app/.env found - the image is public"
else
  pass "no .env in /app"
fi

echo "== gate 5: inference contract on real images =="
if [ ! -d "$SAMPLES" ]; then
  note "SAMPLES=$SAMPLES not found - skipping timing/format gate"
  note "create ./samples with a .png, .jpg and .tiff, then rerun"
else
  cid=$(docker run -d --device /dev/kfd --device /dev/dri --shm-size=16g "$IMAGE" sleep infinity 2>/dev/null)
  if [ -z "$cid" ]; then
    fail "container would not start (check ROCm device flags on this host)"
  else
    docker exec "$cid" mkdir -p /app/input /app/output
    for f in "$SAMPLES"/*; do
      ext="${f##*.}"
      case "$ext" in png|jpg|jpeg|tif|tiff) ;; *) continue ;; esac
      name=$(basename "$f")
      docker cp "$f" "$cid:/app/input/$name" >/dev/null 2>&1
      start=$(date +%s)
      docker exec "$cid" python3 /app/app.py --input-image "/app/input/$name" >/dev/null 2>&1
      end=$(date +%s)
      took=$((end - start))
      out="/app/output/${name%.*}_output.json"
      if [ "$took" -gt 30 ]; then
        fail "$name took ${took}s (per-image budget is 30s)"
      elif ! docker exec "$cid" test -f "$out" >/dev/null 2>&1; then
        fail "$name produced no $out"
      else
        body=$(docker exec "$cid" cat "$out")
        if printf '%s' "$body" | python3 -c 'import json,sys; d=json.load(sys.stdin); assert isinstance(d.get("text"),str); c=d.get("confidence"); assert isinstance(c,(int,float)) and 0.0<=c<=1.0' 2>/dev/null; then
          pass "$name ${took}s -> $body"
        else
          fail "$name bad schema: $body"
        fi
      fi
    done
    echo "  peak VRAM observed during the run above (harness samples every 3s, must be 1-48 GiB):"
    note "docker exec $cid rocm-smi --showmeminfo vram --watch  (or amd-smi metric --mem)"
    docker rm -f "$cid" >/dev/null 2>&1
  fi
fi

echo
if [ "$fails" -eq 0 ]; then
  echo "preflight clean - but you still have not verified VRAM, so watch that during a full run"
else
  echo "$fails gate(s) failed - fix before submitting"
fi
exit "$fails"
