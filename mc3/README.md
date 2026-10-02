# AMD AI Academy — Mini-Challenge 3 (Retrieval-Augmented Generation)

A container that answers questions about a folder of mixed documents, and names
the files each answer came from. Graded on exactly two things: the answer, and
the citation. There is no partial credit — a correct answer with an incomplete or
over-long citation list scores zero for that question.

## Scoring contract

The harness makes **two** kinds of invocation, and the difference decides whether
this scores at all.

```
python3 /app/app.py --index /app/corpus                    # once, at startup

python3 /app/app.py --corpus /app/corpus \
  --query-id query_01 --query "What is the maximum junction temperature of the TQ-40?"
```

Each question is a **separate process** writing `/app/output/<query-id>_output.json`:

```json
{ "answer": "94", "citations": ["specs/tq40_datasheet_r2.pdf"], "confidence": 0.91 }
```

`answer` and `citations` must be present on **every** response, including
refusals. A missing key is malformed and scores zero — the graders do not infer a
refusal from a missing field, because a crashed writer and a considered refusal
would then look identical.

| Gate | Requirement |
|---|---|
| Base image | final stage `rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0`, verified by layer identity |
| Image size | ≤ 60 GiB uncompressed |
| Startup | container start + model load + `--index` ≤ 10 min |
| Per question | 30 s per `--query` |
| Total | 10 min for all 10 questions |
| VRAM | peak between 1 GiB and 48 GiB, sampled continuously |

## Layout

```
Dockerfile                 mandated ROCm base, --no-deps installs, ROCm verified at build
app.py                     thin client: socket request, atomic JSON write, never fails loudly
daemon.py                  long-running: model, BM25 index, image captioning, answer loop
corpus.py                  defensive multi-format walk (pdf/docx/xlsx/csv/code/images)
wire.py                    unix socket transport
requirements.txt           full transitive set, installed with --no-deps
preflight.sh               replicates the graders' checks, incl. --cap-drop DAC_OVERRIDE
tests/test_mc3.py          43 structural checks (socket, output schema, refusals, walk)
tests/test_mc3_recall.py   retrieval accuracy against known gold files
tests/test_mc3_encrypted.py encrypted PDFs must yield nothing, whatever the password
```

## Architecture

The brief's first named trap: *your script is a new process for every question.* A
model loaded inside `app.py` loads ten times; a corpus indexed there re-parses
every PDF, spreadsheet and image ten times. Either blows the 30 s budget on every
question and a perfectly good model scores zero.

So all state lives in one long-running process:

```
  --index ──┐                        ┌── model  (Qwen2.5-VL-7B-Instruct, fp32)
            ├──► daemon.py ── BM25 ──┤
  --query ──┘     (unix socket)      └── index  (incl. VLM captions of images)
```

`app.py` is a marshalling layer. It holds no model, opens no corpus, and writes
its JSON atomically (temp file + `os.replace`) so a partial write can never be
graded. If the socket is unreachable it still writes a well-formed empty response,
because a malformed file scores zero and a valid refusal does not.

`CMD` starts the daemon. The autoload thread indexes in the background so the
socket is up immediately, and `ensure_indexed()` makes a later `--index` request a
no-op rather than a second pass.

## The ROCm constraint that shapes everything

On ROCm, the Qwen2.5-VL **vision tower segfaults** (SIGSEGV) inside `model.visual()`
when weights are bf16 or fp16 — before any generation happens. `attn_implementation="eager"`
is also required; the SDPA kernel crashes on the same shapes.

```python
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_ID, dtype=torch.float32, attn_implementation="eager")
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
```

This is not an optimisation choice, it is the only configuration that runs. It
costs memory — 7B in fp32 is ~28 GiB of weights before a single token — which is
why the 48 GiB VRAM gate is the binding constraint on this whole solution.

`device_map="auto"` is deliberately **not** used. Dropping it without replacing it
leaves the model on CPU, where a 7B fp32 model generates at ~1-3 tokens/sec: it
looks like a hang rather than an error, and the log still says `model ready`.
Placement is therefore explicit and logged, and a `WARNING` fires if no CUDA
device is present.

## Citation discipline

Citations are compared as an **exact set**. Returning the whole retrieval fails
almost every question even when the answer is right. The test is *necessity*, not
relevance: cite a file only if removing it would make the answer impossible.

So the model names its sources, and then they are filtered:

```python
known = {p.lower(): p for p in grouped}
citations = [known[c.lower()] for c in cites if c.lower() in known]
```

Nothing is padded with the rest of the retrieval, and hallucinated paths are
dropped. Retrieval returns 40 chunks because recall matters; **citation is decided
separately** by the model plus that filter. Measured: seven documents all
discussing junction temperature, one containing the value → one citation.

Multi-hop questions count every step of the chain, so the prompt permits citing
more than one file when each was genuinely load-bearing.

## Unanswerable questions

Graded as `{"answer": "", "citations": [], "confidence": 0.0}`. Two distinct
mechanisms enforce this:

- The prompt instructs the model to reply `NONE` when the sources do not contain
  the answer. `NONE` maps to an empty answer with no citations.
- **Encrypted files are never decrypted.** The corpus deliberately contains an
  encrypted PDF whose contents include the answer to a question scored as
  *unanswerable* — a file you cannot open is not a source. Calling
  `reader.decrypt("")` appears safe but succeeds whenever the file was encrypted
  with an empty user password, which is common, and feeds the secret into
  retrieval. `parse_pdf()` refuses every encrypted file unconditionally.

## Corpus handling

The folder is built to break naive implementations, so every step is defensive:
an empty directory, an unknown file type, a `chmod 000` file, and an encrypted
PDF. One raising parser must not abort the walk — because walk order follows the
directory listing, which files get indexed then depends on filename order. That is
how a submission scores well in testing and badly on the graded set.

DOCX and PPTX are parsed with `zipfile` + `ElementTree`, so the walk has **no
dependency** on `python-docx` or `python-pptx`. Every optional parser
(`pypdf`, `openpyxl`) is imported inside a `try`, so a missing one degrades that
file type instead of killing the module import.

Images hold no text to index, and two graded questions are answerable *only* from
text printed inside an image. Each is captioned through the vision tower during the
index pass and stored as an ordinary searchable chunk, so it still cites
correctly. Caption length is capped at 256 tokens (`MC3_IMAGE_TOKENS`): the vision
prompt asks for exact transcription, not prose, and at 900 tokens a single image
took minutes inside the startup budget.

A caption failure is **loud**, because it is otherwise invisible:

```
WARNING 1/2 images could not be described and are unretrievable: ['backplane_pinout.png']
```

## ROCm torch protection

Installing almost any torch-dependent package lets pip resolve a CUDA build over
the mandated ROCm one. It fails at runtime as
`RuntimeError: operator torchvision::nms does not exist`, which never mentions
torch. Every install is `--no-deps` with the full transitive set listed by hand,
and the Dockerfile verifies ROCm as a build step:

```dockerfile
RUN pip install --no-cache-dir --no-deps -r /app/requirements.txt \
    && python3 -c "import sys,torch; sys.exit(1) if not torch.version.hip else None"
```

`transformers`, `tokenizers` and `huggingface_hub` are **pinned** to the verified
4.57 line. Unpinned, the build resolves to transformers 5.x and huggingface_hub
2.x — two major versions newer — and because `--no-deps` suppresses resolution,
that surfaces as an opaque processor or dtype error at model load rather than at
install time.

## Verification

Measured on an AMD notebook session, ROCm 6.4 and 7.0, MI300X:

| Limit | Cap | Worst measured | Margin |
|---|---|---|---|
| VRAM peak | 48 GiB | **39.7 GiB** | 8.3 GiB |
| Per question | 30 s | **1.58 s** | 28x |
| Startup (load + index) | 600 s | ~10 s | — |
| All 10 questions | 600 s | ~16 s | — |

Worst case was measured deliberately, against a 33 k-character corpus built so
retrieval fills the full `MAX_CONTEXT_CHARS` budget, since attention memory scales
with context squared.

Correctness confirmed on hardware: text retrieval, identifier lookup
(`4471-B` → `14 days`), **a question answered only by an image**
(`THERM_ALERT#` → `specs/backplane_pinout.png`), unanswerable → empty, and ROCm
intact after every install. `tests/test_mc3_recall.py` holds retrieval at 7/7 gold
files at rank 1.

## Defects found and fixed during development

Recorded because the silent ones are the interesting ones.

| Defect | Effect had it shipped |
|---|---|
| bf16 in the model loader | **SIGSEGV on ROCm** — the Mini-Challenge 2 crash, inherited unnoticed because MC3 had never run on hardware |
| `device_map` dropped during that fix | Model silently on **CPU**: minutes per image, empty answers, while logging `model ready` |
| Autoload and `--index` both indexing | Every run captioned every image **twice**, inside the startup budget |
| `decrypt("")` on encrypted PDFs | **Leaked** the contents of any file encrypted with an empty user password — the exact unanswerable-question trap |
| Unpinned `transformers` | Build resolves to 5.x / hub 2.x, failing opaquely at model load |
| Silent caption failure | Images unretrievable with no error anywhere; every image question scores zero |
| No retrieval-accuracy test | 43 structural tests, none checking that retrieval found the right file |

## Known limitations

Stated plainly, because they are the real risk surface.

- **The container has not been built.** Docker was unavailable in the development
  environment, so everything above was measured in the notebook session (ROCm 6.4,
  torch 2.9) while the Dockerfile targets ROCm 10.0 / torch 2.13 / Python 3.14.
  All cp314 wheels were confirmed to exist remotely, so the interpreter version
  should not break the build, but the image itself is unverified.
- **Weights are fetched on first boot**, inside the 600 s startup budget. If the
  evaluation environment has no network access this scores zero overall. Baking
  ~16 GB into the image would remove that dependency but pushes a ~41-46 GiB image
  against the 60 GiB cap, and neither option could be tested without Docker.
- **Tested against a synthetic 3-file corpus**, not the published sample kit.
  Recall and the citation filter were verified against known gold files, but corpus
  variety beyond PDF/PNG/TXT is untested on hardware.

## Running the checks

```bash
# structural + accuracy + encrypted-PDF suites
python3 tests/test_mc3.py
python3 tests/test_mc3_recall.py
python3 tests/test_mc3_encrypted.py

# replicate the grading pipeline against a built image
docker run -d --name mc3 --cap-drop DAC_OVERRIDE mc3-rag:v1 sleep infinity
docker exec mc3 /app/preflight.sh
```

`--cap-drop DAC_OVERRIDE` is required, not optional: running as root bypasses mode
bits, so `chmod 000` does not actually make a file unreadable unless that
capability is dropped — which is what the graders do.
