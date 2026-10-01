"""Generates the solo-team cover image.

    python scripts/make_cover.py "ChudsonAI"
    python scripts/make_cover.py "SignScope" out/cover.png
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
BG_TOP = (12, 12, 16)
BG_BOTTOM = (28, 28, 36)
RED = (237, 28, 36)
DIM = (150, 152, 162)
LIGHT = (240, 241, 245)
FONT_DIR = r"C:\Windows\Fonts"


def font(name, size):
    for candidate in (name, "segoeuib.ttf", "arialbd.ttf", "arial.ttf"):
        path = os.path.join(FONT_DIR, candidate)
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def gradient():
    img = Image.new("RGB", (W, H), BG_TOP)
    d = ImageDraw.Draw(img)
    for y in range(H):
        t = y / (H - 1)
        d.line(
            [(0, y), (W, y)],
            fill=tuple(int(BG_TOP[i] + (BG_BOTTOM[i] - BG_TOP[i]) * t) for i in range(3)),
        )
    return img


def plate(d, box):
    d.rounded_rectangle(box, radius=10, outline=(64, 66, 78), width=3)
    d.rounded_rectangle(
        (box[0] + 8, box[1] + 16, box[2] - 8, box[3] - 16), radius=6, outline=(44, 46, 56), width=2
    )


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ChudsonAI"
    out = sys.argv[2] if len(sys.argv) > 2 else "cover.png"

    img = gradient()
    d = ImageDraw.Draw(img, "RGBA")

    d.rectangle((0, 0, 10, H), fill=RED)

    for box in ((880, 120, 1140, 250), (900, 300, 1150, 420)):
        plate(d, box)
    d.text((900, 160), "7ABC123", font=font("arialbd.ttf", 34), fill=(70, 72, 86))
    d.text((920, 330), "STOP", font=font("arialbd.ttf", 30), fill=(70, 72, 86))

    d.text((70, 130), "AMD AI ACADEMY CHALLENGE", font=font("segoeuib.ttf", 22), fill=RED)
    d.text((70, 200), name, font=font("segoeuib.ttf", 78), fill=LIGHT)
    d.text((70, 310), "Vision-language OCR on AMD ROCm", font=font("segoeuib.ttf", 34), fill=DIM)
    d.text(
        (70, 360),
        "License plates and road signage, under glare, blur and low light",
        font=font("segoeui.ttf", 24),
        fill=(110, 112, 122),
    )

    chips = ["ROCm", "Qwen2.5-VL", "Open Source", "Solo Builder"]
    x = 70
    for chip in chips:
        w = d.textlength(chip, font=font("segoeuib.ttf", 20)) + 32
        d.rounded_rectangle((x, 470, x + w, 512), radius=21, outline=(58, 60, 70), width=2)
        d.text((x + 16, 480), chip, font=font("segoeuib.ttf", 20), fill=(178, 180, 190))
        x += w + 14

    d.text((70, 560), "lablab.ai  ·  Sep 1 - Dec 1, 2026", font=font("segoeui.ttf", 20), fill=(96, 98, 108))

    parent = os.path.dirname(os.path.abspath(out))
    if parent:
        os.makedirs(parent, exist_ok=True)
    img.save(out, "PNG", optimize=True)
    print(f"wrote {out} ({W}x{H}, {os.path.getsize(out) // 1024} KB)")


if __name__ == "__main__":
    main()
