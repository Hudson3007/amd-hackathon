# syntax=docker/dockerfile:1
#
# Mini-Challenge 2 submission image.
#
# HARD RULE: the final stage must start from the mandated ROCm base. AMD verifies
# this by layer identity, so never use `docker build --squash`, buildah, or a
# `FROM scratch` consolidation. Put extra deps in /app/requirements.txt instead.
# The FIRST build on this machine starts from the base, so do not reorder stages.

ARG ROCM_BASE=rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0
ARG PY_BASE=python:3.12-slim
ARG MODEL_ID=Qwen/Qwen2.5-VL-7B-Instruct
ARG HF_BASE=https://huggingface.co

# ---------------------------------------------------------------- Stage 0
# Fetch weights here so they never bloat the ROCm image's build context.
FROM ${PY_BASE} AS weights
ARG MODEL_ID
ARG HF_BASE
ENV HF_HUB_DISABLE_TELEMETRY=1
RUN python3 -m pip install --no-cache-dir "huggingface_hub[hf_transfer]>=0.27"
RUN python3 - <<'PY'
import os
from huggingface_hub import snapshot_download

repo = os.environ["MODEL_ID"]
snapshot_download(
    repo_id=repo,
    revision="main",
    local_dir="/models",
    max_workers=8,
    ignore_patterns=["*.pth", "*.bin", "*.onnx", "*.gguf", "original/*", "*.msgpack"],
)
print("weights staged in /models for", repo)
PY

# ---------------------------------------------------------------- Final stage
FROM ${ROCM_BASE} AS final
ARG MODEL_ID
ENV MODEL_ID=/models \
    OCR_INPUT_DIR=/app/input \
    OCR_OUTPUT_DIR=/app/output \
    OCR_MAX_PIXELS=1254400 \
    OCR_MAX_NEW_TOKENS=48 \
    OCR_DTYPE=float32 \
    OCR_ATTN=eager \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_VERBOSITY=error \
    TOKENIZERS_PARALLELISM=false \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    MPLBACKEND=Agg

RUN mkdir -p /app/input /app/output /models

COPY app/requirements.txt /app/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Fail the build, not the submission, if a CPU/CUDA wheel displaced ROCm torch.
RUN python3 -c "\
import sys, torch; \
hip = getattr(torch.version, 'hip', None); \
print('torch', torch.__version__, 'hip', hip); \
sys.exit(0 if hip else 'ROCm torch missing - a non-ROCm wheel replaced it')"

COPY app/ /app/
COPY --from=weights /models/ /models/

WORKDIR /app
CMD ["python3", "/app/app.py"]
