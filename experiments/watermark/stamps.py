"""Test-only watermarks (never used in training, see PREREG.md).

S1: the shutterstock stamp of the first stamp test (dark bottom banner with name and number + one big faint diagonal word).
    The layouts exist in training (banner, diagonal), the name does not.
S2: a faint "alamy" word repeated in a tilted grid over the whole photo. Neither the layout nor the name is in training.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, max(8, size))
    except OSError:
        return ImageFont.load_default(size=max(8, size))


def s1(image: Image.Image) -> Image.Image:
    im = image.convert("RGB").copy(); w, h = im.size; d = ImageDraw.Draw(im, "RGBA"); bh = max(18, h // 12)
    d.rectangle([0, h - bh, w, h], fill=(20, 20, 20, 255))
    d.text((8, h - bh + bh // 4), "shutterstock · 1768995167", fill=(255, 255, 255, 255), font=_font(max(10, bh // 2)))
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((w // 8, h // 2 - w // 16), "shutterstock", fill=(255, 255, 255, 110), font=_font(max(16, w // 9)))
    return Image.alpha_composite(im.convert("RGBA"), layer.rotate(20, center=(w // 2, h // 2))).convert("RGB")


def s2(image: Image.Image) -> Image.Image:
    im = image.convert("RGB"); w, h = im.size
    size = max(12, min(w, h) // 12)
    big = Image.new("RGBA", (2 * w, 2 * h), (0, 0, 0, 0)); d = ImageDraw.Draw(big); f = _font(size)
    step_x, step_y = int(size * 4.5), int(size * 2.6)
    for row, y in enumerate(range(0, 2 * h, step_y)):
        for x in range(-(row % 2) * step_x // 2, 2 * w, step_x):
            d.text((x, y), "alamy", fill=(255, 255, 255, 90), font=f)
    big = big.rotate(-30, center=(w, h))
    layer = big.crop((w // 2, h // 2, w // 2 + w, h // 2 + h))
    return Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")


STAMPS = {"S1": s1, "S2": s2}
