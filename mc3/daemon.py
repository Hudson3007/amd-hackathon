"""Long-running server for Mini-Challenge 3.

The evaluation harness execs /app/app.py once per question, so loading a model
inside that script would load it ten times and blow the 30-second budget. This
daemon is started by the container CMD, holds the parsed corpus, the retrieval
index and the models, and answers every request over a unix socket.

Retrieval is BM25 first. The graded questions turn on exact identifiers - part
numbers, error codes, pin names, firmware revisions, board revisions - and
lexical scoring is stronger than embeddings on those, while also adding zero
dependencies, which keeps pip from swapping the mandated ROCm torch build for a
CUDA one.
"""

import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import wire  # noqa: E402
from corpus import index_corpus  # noqa: E402

CORPUS_DIR = os.environ.get("MC3_CORPUS", "/app/corpus")
MODEL_ID = os.environ.get("MC3_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
RETRIEVE_TOP_K = int(os.environ.get("MC3_TOP_K", "40"))
MAX_CONTEXT_CHARS = 24000
CACHE_DIR = os.environ.get("MC3_CACHE", "/app/.cache")
IMAGE_CONTEXT = 900

VISION_PROMPT = """Describe this image for a search index.

Transcribe EVERY piece of printed text exactly as it appears, including labels,
part numbers, pin names, codes, revisions and captions. Keep identifiers whole
and do not summarise them away.

Then add one or two sentences of context: what the image is, and what it depicts.

Answer with the transcription and context only. No preamble, no questions."""

STOPWORDS = set(
    """a an the is are was were be been being of in on at to for from by with and or but if then
    what which who whom whose when where why how does do did done has have had will would shall
    should can could may might must this that these those there here it its as not no nor so than
    about into over under again further once all any both each few more most other some such only
    own same too very s t just don now i you he she we they them his her their our your my me us
    value values number numbers maximum max tell show give find list many much long please
    """.split()
)


def log(msg):
    print(f"[mc3] {msg}", flush=True)


def tokenize(text):
    return [t for t in re.split(r"[^0-9a-z]+", text.lower()) if t and t not in STOPWORDS]


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #

def chunk_document(doc, target=900, overlap=150):
    """Split a document into overlapping windows, keeping the relpath on each.

    Overlap matters here: a datasheet row can be split across a boundary and the
    value lands alone in the next chunk with no context.
    """
    text = doc.text
    if len(text) <= target:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = min(start + target, len(text))
        if end < len(text):
            window = text[start:end]
            pivot = window.rfind("\n", start + target // 2)
            if pivot > start:
                end = pivot + 1
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return [c for c in chunks if c.strip()]


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #

class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.docs = documents
        self.freqs = [self._counts(d["text"]) for d in documents]
        self.lengths = [sum(f.values()) for f in self.freqs]
        self.avg = sum(self.lengths) / max(1, len(self.lengths))
        df = {}
        for f in self.freqs:
            for term in f:
                df[term] = df.get(term, 0) + 1
        n = max(1, len(self.freqs))
        self.idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    @staticmethod
    def _counts(text):
        counts = {}
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            counts[tok] = counts.get(tok, 0) + 1
        return counts

    def scores(self, query_tokens):
        out = [0.0] * len(self.docs)
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in enumerate(self.freqs):
                f = freq.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg)
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.bm25 = None
        self.chunks = []
        self.images = []
        self.stats = {}
        self.root = CORPUS_DIR
        self.model = None
        self.processor = None
        self.ready = False
        self.index_ready = threading.Event()


STATE = State()


def build_index(root):
    started = time.time()
    log(f"indexing {root}")
    docs, images, stats = index_corpus(root)
    log(f"parsed {len(docs)} text docs, {len(images)} images, stats={stats}")

    # describe_image resolves image paths against STATE.root, so it has to point
    # at this corpus before the image loop runs.
    with STATE.lock:
        STATE.root = root

    chunks = []
    for doc in docs:
        for piece in chunk_document(doc):
            chunks.append({"text": piece, "relpath": doc.relpath, "kind": doc.kind})

    # Images hold no text to index, and two of the graded questions are answered
    # only by text printed inside them. Describing each image here, during the
    # index pass, turns them into ordinary searchable text that still cites
    # correctly. This runs against the startup budget, not the per-question one.
    for image in images:
        description = describe_image(image.relpath)
        if description:
            chunks.append(
                {
                    "text": f"[image description]\n{description}",
                    "relpath": image.relpath,
                    "kind": "image",
                }
            )

    with STATE.lock:
        STATE.chunks = chunks
        STATE.images = images
        STATE.root = root
        STATE.stats = stats
        STATE.bm25 = BM25(chunks) if chunks else None

    log(f"index built: {len(chunks)} chunks in {time.time() - started:.1f}s")
    STATE.index_ready.set()


def retrieve(query, top_k=RETRIEVE_TOP_K):
    with STATE.lock:
        chunks = list(STATE.chunks)
        images = list(STATE.images)
        bm25 = STATE.bm25
    if not chunks or bm25 is None:
        return [], images

    toks = tokenize(query)
    raw = bm25.scores(toks)
    if not any(raw):
        return [], images

    # Identifier boost: a query containing a part-number-like token should rank
    # chunks containing that exact token far above lexical neighbours.
    idents = [t for t in re.findall(r"\b[a-z]*\d[\w.\-/]*\b", query.lower()) if len(t) > 2]
    if idents:
        for i, chunk in enumerate(chunks):
            low = chunk["text"].lower()
            for ident in idents:
                if ident in low:
                    raw[i] += 12.0 * idf_boost(bm25, ident)

    best = max(raw) or 1.0
    ranked = sorted(range(len(raw)), key=lambda i: raw[i], reverse=True)
    return [chunks[i] for i in ranked[:top_k]], images


def idf_boost(bm25, term):
    return min(3.0, bm25.idf.get(term, 1.0))


def group_by_file(chunks):
    grouped = {}
    for chunk in chunks:
        grouped.setdefault(chunk["relpath"], []).append(chunk["text"])
    return grouped


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #

def _cache_path(relpath):
    digest = hashlib.sha256(relpath.encode("utf-8")).hexdigest()[:20]
    return os.path.join(CACHE_DIR, "image_descriptions", f"{digest}.txt")


def _cache_get(relpath):
    try:
        with open(_cache_path(relpath), encoding="utf-8") as fh:
            cached = fh.read().strip()
        return cached or None
    except OSError:
        return None


def _cache_put(relpath, text):
    target = _cache_path(relpath)
    try:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        tmp = f"{target}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, target)
    except OSError as exc:
        log(f"could not cache description for {relpath}: {exc}")


def describe_image(relpath):
    """Return searchable text for an image, cached across re-index runs.

    An image that cannot be described is still cited correctly if it appears in
    a text answer, but it will not be retrievable on its own, so the failure is
    logged rather than raised.
    """
    cached = _cache_get(relpath)
    if cached:
        return cached

    if STATE.model is None or STATE.processor is None:
        return None

    path = os.path.join(STATE.root, relpath) if STATE.root else relpath
    try:
        from PIL import Image

        picture = Image.open(path).convert("RGB")
    except Exception as exc:
        log(f"cannot open image {relpath}: {exc}")
        return None

    try:
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ]
        prompt = STATE.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = STATE.processor(text=[prompt], images=[picture], return_tensors="pt").to(
            STATE.model.device
        )
        with torch.inference_mode():
            out = STATE.model.generate(
                **inputs,
                max_new_tokens=IMAGE_CONTEXT,
                do_sample=False,
                pad_token_id=STATE.processor.tokenizer.eos_token_id,
            )
        trimmed = out[0][inputs["input_ids"].shape[1]:]
        text = STATE.processor.decode(trimmed, skip_special_tokens=True).strip()
    except Exception as exc:
        log(f"vision call failed for {relpath}: {exc}")
        return None

    if text:
        _cache_put(relpath, text)
    return text or None


def load_model():
    if os.environ.get("MC3_NO_MODEL") == "1":
        log("MC3_NO_MODEL=1, starting retrieval-only")
        return
    try:
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except Exception as exc:
        log(f"transformers unavailable, running retrieval-only: {exc}")
        return

    try:
        log(f"loading {MODEL_ID}")
        STATE.processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        STATE.model = model
        log("model ready")
    except Exception as exc:
        log(f"model load failed, running retrieval-only: {exc}")
        traceback.print_exc()


ANSWER_PROMPT = """You answer questions about a technical document corpus.

Rules:
- Reply with ONLY the value: a number, part number, version, revision, quarter, pin, or code.
- No sentence, no explanation, no restatement of the question.
- Keep qualifiers that identify the value: "Q3 FY27" not "Q3", "REV-C2" not "C2".
- If the sources do not contain the answer, reply exactly: NONE
- If answering needs two files, cite both.

SOURCES
{context}

QUESTION
{question}

Reply in this format and nothing else:
ANSWER: <value or NONE>
CITES: <comma separated file paths, or NONE>"""


def build_context(grouped):
    parts = []
    for i, (path, pieces) in enumerate(sorted(grouped.items()), start=1):
        body = "\n".join(pieces)[: MAX_CONTEXT_CHARS // max(1, len(grouped))]
        parts.append(f"[{i}] {path}\n{body}")
    return "\n\n".join(parts)[:MAX_CONTEXT_CHARS]


def call_model(prompt, timeout=120):
    if STATE.model is None:
        return None
    import torch

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = STATE.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = STATE.processor(text=[text], return_tensors="pt").to(STATE.model.device)
    with torch.inference_mode():
        out = STATE.model.generate(
            **inputs, max_new_tokens=256, do_sample=False,
            pad_token_id=STATE.processor.tokenizer.eos_token_id,
        )
    trimmed = out[0][inputs["input_ids"].shape[1]:]
    return STATE.processor.decode(trimmed, skip_special_tokens=True)


ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.*)", re.I)
CITES_RE = re.compile(r"CITES\s*:\s*(.*)", re.I)


def parse_model_reply(reply):
    answer, cites = "", []
    if not reply:
        return answer, cites
    m = ANSWER_RE.search(reply)
    if m:
        answer = m.group(1).strip().splitlines()[0].strip()
    c = CITES_RE.search(reply)
    if c:
        raw = c.group(1).strip()
        if raw.upper() != "NONE":
            cites = [p.strip() for p in raw.split(",") if p.strip() and p.strip().upper() != "NONE"]
    return answer, cites


def answer_question(question):
    chunks, images = retrieve(question)
    grouped = group_by_file(chunks)
    if not grouped:
        return {"answer": "", "citations": [], "confidence": 0.0}

    context = build_context(grouped)
    try:
        reply = call_model(ANSWER_PROMPT.format(context=context, question=question))
    except Exception as exc:
        log(f"model call failed: {exc}")
        reply = None

    answer, cites = parse_model_reply(reply)
    if answer.upper() == "NONE" or not answer:
        return {"answer": "", "citations": [], "confidence": 0.0}

    # Citations are scored as an exact set, so keep only the ones the model named
    # that also exist in the corpus, and never pad with the rest of the retrieval.
    known = {p.lower(): p for p in grouped}
    citations = [known[c.lower()] for c in cites if c.lower() in known]
    citations = list(dict.fromkeys(citations))

    return {"answer": answer, "citations": citations, "confidence": 0.75}


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #

class Server:
    """Minimal threaded server over the wire module's transport."""

    def __init__(self):
        self.sock = wire.server_listen()

    def serve_forever(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                continue
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def handle(self, conn):
        with conn:
            buf = bytearray()
            try:
                while b"\n" not in buf:
                    chunk = conn.recv(65536)
                    if not chunk:
                        return
                    buf.extend(chunk)
                payload = json.loads(bytes(buf).split(b"\n", 1)[0].decode("utf-8"))
                conn.sendall((json.dumps(dispatch(payload)) + "\n").encode("utf-8"))
            except Exception:
                traceback.print_exc()


def dispatch(payload):
    op = payload.get("op")
    try:
        if op == "ping":
            return {"ok": True, "ready": STATE.ready, "indexed": STATE.index_ready.is_set()}
        if op == "index":
            build_index(payload.get("root") or CORPUS_DIR)
            return {"ok": True, "stats": STATE.stats}
        if op == "query":
            return answer_question(payload.get("query", ""))
        return {"error": f"unknown op {op}"}
    except Exception as exc:
        traceback.print_exc()
        return {"answer": "", "citations": [], "confidence": 0.0, "error": str(exc)}


def main():
    load_model()

    # The harness runs the container down before it execs --index. Reindex
    # anyway so a stale index can never be served.
    threading.Thread(target=_safe_autoload, daemon=True).start()

    server = Server()
    STATE.ready = True
    log(f"listening on {wire.describe()}")
    server.serve_forever()


def _safe_autoload():
    try:
        if os.path.isdir(CORPUS_DIR):
            time.sleep(1.0)
            build_index(CORPUS_DIR)
    except Exception as exc:
        log(f"autoload failed: {exc}")


if __name__ == "__main__":
    main()


