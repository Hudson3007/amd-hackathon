"""End-to-end checks for the Mini-Challenge 3 pipeline.

    python tests/test_mc3.py [corpus_dir]

Verifies the things the spec says will score you zero if they break: the walk
survives the hostile files, the client/daemon contract holds, and retrieval puts
the file a question actually needs at the top.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MC3 = os.path.join(os.path.dirname(HERE), "mc3")
CORPUS = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(tempfile.gettempdir(), "mc3_corpus")
)

sys.path.insert(0, MC3)

PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{f'  -> {detail}' if detail and not ok else ''}")


# --------------------------------------------------------------------------- #
print("\n[1] corpus walk survives the hostile files")
from corpus import index_corpus, walk_corpus  # noqa: E402

all_files = list(walk_corpus(CORPUS))
docs, images, stats = index_corpus(CORPUS)
indexed = {d.relpath for d in docs} | {i.relpath for i in images}

check("walk does not abort", len(all_files) >= 12, f"only {len(all_files)} files seen")
check("empty directory tolerated", os.path.isdir(os.path.join(CORPUS, "archive")) and "archive" not in indexed)
check("unknown binary type skipped", "vendor/telemetry_capture.dat" not in indexed)
check("encrypted pdf yields no text", "vendor/supplier_agreement_ENCRYPTED.pdf" not in indexed)
check("readable pdf indexed", "specs/tq40_datasheet_r2.pdf" in indexed)
check("docx indexed", "planning/roadmap_fy27.docx" in indexed)
check("xlsx indexed", "support/rma_parts.xlsx" in indexed)
check("csv indexed", "support/bug_database.csv" in indexed)
check("python indexed", "engineering/ingest_service.py" in indexed)
check("log indexed", "logs/prod_inference_2026-09-02.log" in indexed)
check("images deferred to vision", {"specs/backplane_pinout.png", "support/asset_label.jpg"} <=
      {i.relpath for i in images})
check("files after the bad ones still indexed",
      "vendor/internal_audit.txt" in indexed and "support/asset_label.jpg" in indexed)

# --------------------------------------------------------------------------- #
print("\n[2] retrieval ranks the right file first")
import daemon as D  # noqa: E402

docs, images, stats = index_corpus(CORPUS)
D.STATE.chunks = []
for d in docs:
    for piece in D.chunk_document(d):
        D.STATE.chunks.append({"text": piece, "relpath": d.relpath, "kind": d.kind})
D.STATE.images = images
D.STATE.bm25 = D.BM25(D.STATE.chunks)

EXPECTED = [
    ("What is the maximum junction temperature of the TQ-40?", "specs/tq40_datasheet_r2.pdf"),
    ("In which quarter does the TQ-60 enter customer sampling?", "planning/roadmap_fy27.docx"),
    ("What is the part number of the field-replaceable fan assembly for the TQ-40?",
     "support/rma_parts.xlsx"),
    ("Which firmware version fixed ticket ORR-1847?", "support/bug_database.csv"),
    ("What error code is logged when the thermal throttle engages?",
     "logs/prod_inference_2026-09-02.log"),
    ("What is the default batch timeout, in seconds, in the ingest service?",
     "engineering/ingest_service.py"),
]

for question, want in EXPECTED:
    chunks, _ = D.retrieve(question)
    top = chunks[0]["relpath"] if chunks else None
    ranked = [c["relpath"] for c in chunks[:5]]
    check(f"top hit for '{question[:44]}...'", top == want, f"got {top}, wanted {want}; top5={ranked}")

# the multi-hop question needs both files present
chunks, _ = D.retrieve(
    "The production log shows a thermal throttle incident. Which firmware release fixed the underlying defect?"
)
top5 = {c["relpath"] for c in chunks[:5]}
check("multi-hop surfaces both sources",
      {"logs/prod_inference_2026-09-02.log", "support/bug_database.csv"} <= top5, f"top5={top5}")

# a question with no answer in the corpus must not retrieve a confident source
chunks, _ = D.retrieve("What is the unit price of the TQ-40 at 10,000 unit volume?")
check("unanswerable question does not latch onto the encrypted file",
      "vendor/supplier_agreement_ENCRYPTED.pdf" not in {c["relpath"] for c in chunks[:5]})

# --------------------------------------------------------------------------- #
print("\n[2b] image-only questions are retrievable and citable")


DESCRIPTIONS = [
    "Printed text: B14 THERM_ALERT#\nA backplane pinout diagram showing connector pin assignments.",
    "Printed text: BOARD REV-C2\nAn asset label photograph for a field service unit.",
]


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeProcessor:
    tokenizer = type("T", (), {"eos_token_id": 0})()

    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        return "stub"

    def __call__(self, text=None, images=None, return_tensors=None):
        return FakeInputs(input_ids=type("I", (), {"shape": (1, 4)})())

    def decode(self, tokens, skip_special_tokens=True):
        text = DESCRIPTIONS[min(self.calls, len(DESCRIPTIONS) - 1)]
        self.calls += 1
        return text


class FakeModel:
    device = "cpu"

    def __init__(self):
        self.passes = 0

    def generate(self, **kwargs):
        self.passes += 1
        return type("O", (), {"__getitem__": lambda self, i: [0, 0, 0, 0, 0, 0]})()


tmp_cache = os.path.join(tempfile.gettempdir(), "mc3_cache")
shutil.rmtree(tmp_cache, ignore_errors=True)  # else cached descriptions skip the model
fake = FakeModel()
D.STATE.root = CORPUS
D.STATE.model = fake
D.STATE.processor = FakeProcessor()
D.CACHE_DIR = tmp_cache

built = []
D.build_index(CORPUS)

img_chunks = [c for c in D.STATE.chunks if c["kind"] == "image"]
check("images produced indexed descriptions", len(img_chunks) == 2, f"got {len(img_chunks)}")
check("vision model ran once per image", fake.passes == 2, f"passes={fake.passes}")
check("pinout described with its text",
      any("THERM_ALERT" in c["text"] for c in img_chunks), str(img_chunks))
check("image description cites the image path",
      all(c["relpath"].endswith((".png", ".jpg")) for c in img_chunks))

# cache means a rebuild does not re-run vision
fake.passes = 0
D.build_index(CORPUS)
check("descriptions cached, vision not re-run", fake.passes == 0, f"passes={fake.passes}")

for question, want in [
    ("Which backplane pin carries THERM_ALERT# on the TQ-40?", "specs/backplane_pinout.png"),
    ("What board revision is printed on the asset label?", "support/asset_label.jpg"),
]:
    chunks, _ = D.retrieve(question)
    top = chunks[0]["relpath"] if chunks else None
    check(f"image question routes to the image  ->  '{question[:38]}...'",
          top == want, f"got {top}, wanted {want}")

# text retrieval must still win over image descriptions for text questions
chunks, _ = D.retrieve("What is the maximum junction temperature of the TQ-40?")
check("text questions still outrank image descriptions",
      chunks[0]["relpath"] == "specs/tq40_datasheet_r2.pdf", chunks[0]["relpath"])

D.STATE.model = None
D.STATE.processor = None
D.build_index(CORPUS)

# --------------------------------------------------------------------------- #
print("\n[3] client/daemon contract")
sock = os.path.join(tempfile.gettempdir(), "mc3_test.sock")
outdir = os.path.join(tempfile.gettempdir(), "mc3_out")
env = dict(os.environ, MC3_SOCKET=sock, MC3_OUTPUT=outdir, MC3_NO_MODEL="1", PYTHONPATH=MC3)

server = subprocess.Popen([sys.executable, os.path.join(MC3, "daemon.py")], env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

import wire  # noqa: E402

ready = False
for _ in range(80):
    time.sleep(0.5)
    if server.poll() is not None:
        break
    try:
        sock = wire.client_connect(time.time() + 1.0)
        sock.close()
        ready = True
        break
    except OSError:
        continue

if not ready:
    server.terminate()
    print("  FAIL  daemon never accepted a connection")
    print((server.stdout.read() or "")[-2000:])
    FAIL.append("daemon startup")
else:
    print(f"  PASS  daemon started and accepted a connection on {wire.describe()}")

    def client(*args, timeout=120):
        return subprocess.run([sys.executable, os.path.join(MC3, "app.py"), *args],
                              env=env, capture_output=True, text=True, timeout=timeout)

    r = client("--index", CORPUS)
    check("--index exits 0", r.returncode == 0, r.stderr[-400:])

    r = client("--corpus", CORPUS, "--query-id", "query_01",
               "--query", "What is the maximum junction temperature of the TQ-40?")
    check("--query exits 0", r.returncode == 0, r.stderr[-400:])

    target = os.path.join(outdir, "query_01_output.json")
    check("output file written", os.path.exists(target))
    if os.path.exists(target):
        with open(target, encoding="utf-8") as fh:
            payload = json.load(fh)
        check("answer key present", "answer" in payload)
        check("citations key present", "citations" in payload)
        check("confidence key present", "confidence" in payload)
        check("citations is a list", isinstance(payload.get("citations"), list))
        check("confidence is a number", isinstance(payload.get("confidence"), (int, float)))

    # the harness execs a fresh process per question, so repeat to prove the
    # index survives in the daemon rather than being rebuilt each time
    for i in range(2, 6):
        client("--corpus", CORPUS, "--query-id", f"query_{i:02d}",
               "--query", "Which firmware version fixed ticket ORR-1847?")
    produced = sorted(f for f in os.listdir(outdir) if f.endswith("_output.json"))
    check("one output per query id", len(produced) == 5, f"got {produced}")

    # daemon still up after repeated exec
    alive = server.poll() is None
    check("daemon survived repeated exec", alive)

server.terminate()
try:
    server.wait(timeout=15)
except subprocess.TimeoutExpired:
    server.kill()

# --------------------------------------------------------------------------- #
print("\n[4] failure paths still produce valid JSON")
env2 = dict(env, MC3_SOCKET=os.path.join(tempfile.gettempdir(), "mc3_absent.sock"),
            MC3_OUTPUT=os.path.join(tempfile.gettempdir(), "mc3_out_dead"))
r = subprocess.run([sys.executable, os.path.join(MC3, "app.py"), "--corpus", CORPUS,
                    "--query-id", "query_dead", "--query", "anything"],
                   env=env2, capture_output=True, text=True, timeout=120)
dead = os.path.join(env2["MC3_OUTPUT"], "query_dead_output.json")
check("client survives a dead daemon", r.returncode == 0, r.stderr[-400:])
check("dead-daemon run still writes valid JSON", os.path.exists(dead))
if os.path.exists(dead):
    with open(dead, encoding="utf-8") as fh:
        payload = json.load(fh)
    check("dead-daemon payload is schema valid",
          isinstance(payload.get("answer"), str)
          and isinstance(payload.get("citations"), list)
          and isinstance(payload.get("confidence"), (int, float)), str(payload))

r = subprocess.run([sys.executable, os.path.join(MC3, "app.py")], env=env,
                   capture_output=True, text=True, timeout=60)
check("no-args invocation does not crash", r.returncode == 0, r.stderr[-300:])

r = subprocess.run([sys.executable, os.path.join(MC3, "app.py"), "--corpus", CORPUS,
                    "--query", "no query id given"], env=env, capture_output=True, text=True, timeout=60)
check("missing --query-id does not crash", r.returncode == 0, r.stderr[-300:])

# --------------------------------------------------------------------------- #
print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("failed: " + ", ".join(FAIL))
sys.exit(1 if FAIL else 0)
