# AMD AI Academy — Mini-Challenge 2 (OCR)

Cloud-only target: no local AMD GPU. Development happens on the AMD notebook
session, the container is built and pushed from a disk-large cloud VM.

## Scoring contract

10 hidden images, 20 points each, pass/fail. The harness runs, per image:

```
python3 /app/app.py --input-image /app/input/image_01.png
```

and reads `/app/output/image_01_output.json`:

```json
{ "text": "7ABC123", "confidence": 0.94 }
```

Both sides are normalized before comparison: uppercase, all whitespace removed,
and `- . · _` removed. So only *which characters* you emit matters.

## What scores zero (not deducted — zeroed)

| Gate | Requirement |
|---|---|
| Base image | final stage must be `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`, verified by **layer identity** |
| Image size | ≤ 60 GiB uncompressed |
| Startup | container starts within 10 min |
| Per image | 30 s per inference call |
| Total | 10 min for all 10 images |
| VRAM | peak must sit between 1 GiB and 48 GiB, sampled every 3 s |

Never `docker build --squash`, buildah, or `FROM scratch` consolidation: it
destroys the layer identity AMD verifies and the submission is rejected even
though it was built on the right base.

## Layout

```
Dockerfile                 final stage = mandated ROCm base, weights staged separately
app/app.py                 CLI contract, JSON output, per-image guard, batch mode
app/pipeline.py            model load (once), VLM inference, confidence, OOM fallback
app/normalize.py           transcription rules (the actual scoring logic)
app/requirements.txt       runtime deps
tests/test_normalize.py    24 checks against the 10 published samples
scripts/preflight.sh       replicates all five gates
```

## 1. Develop on the AMD notebook session

`https://notebooks.amd.com/hackathon` — sign in with AMD Dev Program SSO, press
"Launch Notebook".

- **3 h/day**, refreshes every 24 h, unused time does not roll over. An idle
  session still burns quota, so always press "Turn-off Session".
- The redirect URL is single-use and short-lived. Go back to the URL above each
  time instead of bookmarking the session.
- GPUs are first-come first-served; "No GPUs Available" means try later.
- The session image *is* the mandated base, so what you develop against is what
  gets graded.

**Storage is the classic footgun.** 25 GB persists, but only in one directory
depending on your pod name:

| Pod name contains | Persist here |
|---|---|
| `jupyter-hack-***` | `/persistent` |
| `rgapi-hackathon-***` | `/workspace` |

`pip install` does **not** survive a session reset. Re-run it each session.

```bash
python3 -m pip install -r app/requirements.txt
python3 app/app.py --input-dir /path/to/samples --output-dir /path/to/out
python3 tests/test_normalize.py
```

Batch mode is a local convenience; the harness only ever uses `--input-image`.

## 2. Model choice

Default `Qwen2.5-VL-7B-Instruct` (~16 GB bf16). The pool may hand you CDNA or
RDNA hardware, so check what you actually got:

```bash
rocm-smi --showproductname
python3 -c "import torch; print(torch.cuda.get_device_properties(0).total_memory/2**30)"
```

- ≥ 24 GB → stay on 7B.
- 16 GB or less → build with `--build-arg MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct`.

Accuracy on small Chinese characters and multi-line signage is what separates a
180 from a 200, so prefer 7B when the hardware allows.

`normalize.py` is a safety net, not the strategy: the prompt asks for the right
answer and the rules clean up the common failure modes (banner text leaking into
plate output, units appended to advisory plaques, lines not joined).

## 3. Build and push

Building needs no GPU — only disk. The base plus weights is roughly 30 GB, so
give the VM **100 GB**. This machine has no Docker, so use a cloud VM.

```bash
docker build -t YOUR_DOCKERHUB_USER/amd-ocr:mc2 .

# smoke-test before pushing
IMAGE=YOUR_DOCKERHUB_USER/amd-ocr:mc2 SAMPLES=./samples ./scripts/preflight.sh

docker push YOUR_DOCKERHUB_USER/amd-ocr:mc2
```

**Use Docker Hub, not GHCR.** The weights land in a single ~16 GB layer and
GHCR rejects layers over 10 GB. Docker Hub allows it.

Building takes a while (large base pull, then a 16 GB download). That is normal.

## 4. Before submitting

```bash
IMAGE=YOUR_DOCKERHUB_USER/amd-ocr:mc2 SAMPLES=./samples ./scripts/preflight.sh
```

It checks base layer identity, uncompressed size, contract paths, `.env`
absence, and per-image timing plus JSON schema for a PNG, JPEG and TIFF.

Two things it cannot check for you: **peak VRAM** (watch `rocm-smi
--showmeminfo vram` during a full run) and the graded set being harder than the
samples — hardcoding sample answers scores zero.

Submission is the image reference, publicly pullable with no credentials.

- The image is public. No `.env`, API keys, tokens or cloud credentials, ever.
- Keep the reference off public repos — `.gitignore` already excludes
  `submission.env` and `*.image_ref`. Do not announce where the image lives.
- Nothing is built at evaluation time, so what you test is what is scored.

## 5. XP, since that is what actually pays

The $2,500 / $1,500 / $1,000 prizes go to the first three people to reach
**Legend (10,000+ XP)**, not to the best demo. XP updates every Friday.

Highest-yield moves, from the event page:

- **Referrals: 600 XP per friend who fully engages** (50 sign-up + 100
  onboarding + 150 tutorial + 300 project). Enroll, get approved, then
  generate your link. This is the only lever that scales.
- Open-source project +200, complete project +200, build on AMD tech +250,
  project milestones +100 (3 per project).
- Publish on X/LinkedIn tagging `@lablab` + `@AIatAMD` and post the build-in-public
  trail — that is how AMD verifies XP, and it is capped monthly.
- Knowledge Hub: featured by AMD +250, case study +200, tutorial +150.
- Discord: accepted answer +50, contributor badge +250.

Complete the **Build AI** learning pathway for Mini-Challenge 1. Mini-Challenges
3–6 follow the themes RAG/hallucination, a scraping agent, multi-agent software
engineering, and fine-tuning a model to play a novel game — that last one gates
Legend, so leave room for it.
