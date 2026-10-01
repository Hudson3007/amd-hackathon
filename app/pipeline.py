"""ROCm inference for Mini-Challenge 2.

Loads a vision-language model once, transcribes one image, and returns
(finalized_text, confidence). Falls back to a smaller resolution on OOM so a
single oversized test image cannot end the whole run.
"""

import math
import os

from normalize import finalize

PROMPT = """You are a strict OCR engine for vehicle registration plates and road signage.

Read every character visible in the image and return ONLY the transcription.

Rules:
- Return only characters that are printed in the image. No labels, no quotes, no explanation, no units, no added punctuation.
- If the image shows a vehicle license plate, output ONLY the plate number itself. Ignore any state name, province name, motto or slogan printed around the number on the plate.
- On a Chinese license plate, the leading province character and letter ARE part of the registration and must be included.
- If the image shows a road sign, output ALL of its text from top to bottom, joining separate lines with a single space.
- If the sign carries only a number, such as an advisory speed plaque, output only that number.
- For very blurry or dark images, output your single best guess. Never answer that the image is unreadable."""

MODEL_ID = os.environ.get("MODEL_ID", "/models")
MIN_PIXELS = int(os.environ.get("OCR_MIN_PIXELS", 256 * 28 * 28))
MAX_PIXELS = int(os.environ.get("OCR_MAX_PIXELS", 1600 * 28 * 28))
MAX_NEW_TOKENS = int(os.environ.get("OCR_MAX_NEW_TOKENS", 48))

_STATE = {}


def _torch():
    import torch

    return torch


def _is_oom(exc):
    return "outofmemory" in type(exc).__name__.lower()


def _device():
    return "cuda" if _torch().cuda.is_available() else "cpu"


def _dtype():
    # float32 is required, not an optimisation choice: on ROCm 6.4 the Qwen2.5-VL
    # vision tower segfaults (SIGSEGV) when the weights/pixels are bfloat16 or
    # float16. Verified on MI300X: float32+eager runs, bf16+eager crashes in
    # model.visual(). Keep float32 unless a future ROCm base proves it safe.
    return _torch().float32


def _model_class():
    import transformers

    for name in (
        "Qwen2_5_VLForConditionalGeneration",
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
    ):
        cls = getattr(transformers, name, None)
        if cls is not None:
            return cls
    raise RuntimeError("no vision-language model class available in transformers")


def _from_pretrained(cls, path, dtype, device_map):
    # attn_implementation is pinned: the ROCm SDPA kernel segfaults on this
    # vision tower, and eager is the verified-working path.
    kwargs = dict(
        device_map=device_map,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    try:
        return cls.from_pretrained(path, dtype=dtype, **kwargs)
    except TypeError:
        return cls.from_pretrained(path, torch_dtype=dtype, **kwargs)


def load(model_path=None, max_pixels=None):
    """Load model + processor once per process."""
    if "bundle" in _STATE:
        return _STATE["bundle"]

    from transformers import AutoProcessor

    path = model_path or MODEL_ID
    pixels = max_pixels or MAX_PIXELS
    cls = _model_class()
    dtype = _dtype()

    try:
        model = _from_pretrained(cls, path, dtype, "auto")
    except Exception:
        model = _from_pretrained(cls, path, dtype, None)
        model = model.to(_device())

    model.eval()
    processor = AutoProcessor.from_pretrained(
        path, min_pixels=MIN_PIXELS, max_pixels=pixels
    )
    _STATE["bundle"] = (model, processor, pixels)
    return _STATE["bundle"]


def load_image(path):
    from PIL import Image, ImageOps

    with Image.open(path) as im:
        im.load()
        # Phone photos carry EXIF rotation; without this the plate reads sideways.
        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")


def _build_inputs(processor, image):
    try:
        from qwen_vl_utils import process_vision_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        return processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
    except Exception:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return processor(
            text=[text], images=[image], return_tensors="pt", padding=True
        )


def _mean_confidence(out, prompt_len):
    torch = _torch()
    try:
        seq = out.sequences[0][prompt_len:]
        probs = []
        for scores, token_id in zip(out.scores, seq):
            dist = torch.softmax(scores[0].float(), dim=-1)
            probs.append(float(dist[int(token_id)]))
        if not probs:
            return 0.0
        return max(0.0, min(1.0, sum(probs) / len(probs)))
    except Exception:
        return 0.0


def _run_once(model, processor, image):
    inputs = _build_inputs(processor, image)
    inputs = {
        k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()
    }
    prompt_len = inputs["input_ids"].shape[1]
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )
    generated = processor.batch_decode(
        out.sequences[:, prompt_len:], skip_special_tokens=True
    )[0]
    return generated, _mean_confidence(out, prompt_len)


def transcribe(image_path, model_path=None):
    """-> (text, confidence). Never raises."""
    if os.environ.get("OCR_MOCK") == "1":
        return os.environ.get("OCR_MOCK_TEXT", "7ABC123"), 0.9

    image = load_image(image_path)
    try:
        model, processor, _ = load(model_path)
        raw, conf = _run_once(model, processor, image)
    except Exception as exc:
        if os.environ.get("OCR_DEBUG") == "1":
            import traceback

            traceback.print_exc()
        if not _is_oom(exc):
            return "", 0.0
        try:
            _torch().cuda.empty_cache()
            model, processor, pixels = load(model_path)
            shrunk = max(MIN_PIXELS, pixels // 4)
            _STATE["bundle"] = (model, processor, shrunk)
            processor.image_processor.max_pixels = shrunk
            raw, conf = _run_once(model, processor, image)
        except Exception:
            if os.environ.get("OCR_DEBUG") == "1":
                import traceback

                traceback.print_exc()
            return "", 0.0

    text = finalize(raw)
    if not text:
        return "", 0.0
    if not math.isfinite(conf):
        conf = 0.0
    return text, conf
