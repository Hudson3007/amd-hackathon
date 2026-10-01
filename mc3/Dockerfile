# Mini-Challenge 3 - Retrieval-Augmented Generation
#
# Build from this directory:
#   docker build -t mc3-rag:v1 .
#
# The base image is mandated and must stay the first layer, uncompressed, and
# never squashed. Only additional layers are added here.
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.hf \
    MC3_SOCKET=/app/run.sock \
    MC3_CORPUS=/app/corpus \
    MC3_OUTPUT=/app/output \
    MC3_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# Trap 2 from the challenge brief. Installing almost any package that depends on
# torch lets pip resolve that dependency and lay a CUDA build over the mandated
# ROCm one, which then fails at runtime with
# "operator torchvision::nms does not exist" and never mentions torch. Every
# install below is --no-deps, and the full transitive set is listed explicitly
# so nothing is resolved implicitly.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --no-deps -r /app/requirements.txt \
    && python3 - <<'PY'
import sys, torch
if not getattr(torch.version, "hip", None):
    sys.exit(f"ROCm torch was replaced by {torch.__version__}")
print("torch", torch.__version__, "hip", torch.version.hip)
PY

COPY app.py daemon.py wire.py corpus.py /app/
COPY preflight.sh /app/preflight.sh
RUN chmod +x /app/preflight.sh

# Model weights are fetched by the daemon on first boot, not baked in. Nothing
# here exposes a port; the harness execs into the container instead.
RUN mkdir -p /app/output /app/corpus

# The daemon holds the index and the model so that the per-question process
# never pays for a load. Container start runs the model load and the index pass.
CMD ["python3", "/app/daemon.py"]
