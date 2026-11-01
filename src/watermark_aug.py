"""Synthetic stock-photo watermarks, used to break the "watermark -> rotten" shortcut (experiments/watermark).

About half of the rotten photos in this dataset carry a stock-agency or archive watermark, but only about 8% of the fresh
photos do (Valid + Test, checked by eye). A model can then learn "text over the photo -> rotten". Training with synthetic
watermarks stamped more often on fresh photos than on rotten ones makes a watermark equally common in both, so it stops
being evidence for either.

Two strengths: "standard" (experiments/watermark round 1) and "light" (round 2: about half the size, fainter, banners and
number boxes see-through), so the stamp hides less of the vegetable.

Training stamps use the brand words and layouts below. "shutterstock" and "alamy" and the tiled layout are kept out of
training on purpose: experiments/watermark/stamps.py uses them to test on watermarks the model has never seen.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

WORDS = ("dreamstime", "123RF", "depositphotos", "iStock", "Getty Images", "Adobe Stock", "Bigstock", "Pond5",
         "canstockphoto", "fotolia", "PIXTA", "Fotosearch", "VectorStock", "agefotostock", "photo credit")
LAYOUTS = ("banner", "corner", "diagonal", "id_box", "vertical")
FONT_FILES = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
              "/usr/share/fonts/truetype/freefont/FreeSans.ttf")
_FONTS = [f for f in FONT_FILES if Path(f).is_file()]
# (low, high) ranges per strength: sizes as a share of the photo height (diagonal: of the width), opacities 0..1.
# None = fully opaque, as in round 1 (no random draw, so round-1 stamps are reproduced exactly).
STYLES = {
    "standard": {"banner": (0.05, 0.12), "banner_fill": None, "banner_text": None, "corner": (0.03, 0.07),
                 "corner_alpha": (0.6, 1.0), "corner_box": (150 / 255, 170 / 255), "diagonal": (0.08, 0.16),
                 "diagonal_alpha": (0.25, 0.55), "id_box": (0.035, 0.06), "id_fill": None, "vertical": (0.025, 0.045),
                 "vertical_alpha": (0.5, 0.9)},
    "light": {"banner": (0.03, 0.06), "banner_fill": (0.45, 0.75), "banner_text": (0.6, 0.9), "corner": (0.02, 0.04),
              "corner_alpha": (0.35, 0.7), "corner_box": (0.3, 0.5), "diagonal": (0.05, 0.10),
              "diagonal_alpha": (0.12, 0.3), "id_box": (0.02, 0.035), "id_fill": (0.5, 0.8), "vertical": (0.018, 0.03),
              "vertical_alpha": (0.3, 0.6)},
}


def font(size: int, index: int = 0) -> ImageFont.ImageFont:
    size = max(8, int(size))
    if _FONTS:
        return ImageFont.truetype(_FONTS[index % len(_FONTS)], size)
    return ImageFont.load_default(size=size)  # Pillow >= 10.1


class Dice:
    """Random numbers from torch (seeded by set_seed during training, or by an explicit generator)."""

    def __init__(self, generator: torch.Generator | None = None):
        self.generator = generator

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        return low + (high - low) * float(torch.rand(1, generator=self.generator))

    def integer(self, low: int, high: int) -> int:  # low..high inclusive
        return int(torch.randint(low, high + 1, (1,), generator=self.generator))

    def choice(self, items):
        return items[self.integer(0, len(items) - 1)]


def _text_size(draw: ImageDraw.ImageDraw, text: str, typeface) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=typeface)
    return right - left, bottom - top


def _overlay(image: Image.Image, layer: Image.Image) -> Image.Image:
    return Image.alpha_composite(image.convert("RGBA"), layer).convert("RGB")


def stamp(image: Image.Image, dice: Dice | None = None, layouts: tuple[str, ...] = LAYOUTS,
          words: tuple[str, ...] = WORDS, style: str = "standard") -> Image.Image:
    """One or two random training-style watermarks (brand word and/or a photo number)."""
    dice = dice or Dice()
    out = image.convert("RGB")
    for _ in range(dice.integer(1, 2)):
        out = _one(out, dice.choice(layouts), dice, words, STYLES[style])
    return out


def _one(image: Image.Image, layout: str, dice: Dice, words: tuple[str, ...], st: dict) -> Image.Image:
    alpha = lambda key, default=255: default if st[key] is None else int(255 * dice.uniform(*st[key]))
    w, h = image.size
    word = dice.choice(words)
    number = str(dice.integer(1000000, 2099999999))
    face = dice.integer(0, 9)
    light = dice.uniform() < 0.7
    ink = (255, 255, 255) if light else (20, 20, 20)
    layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if layout == "banner":
        bar = max(12 if st["banner_fill"] is None else 8, int(h * dice.uniform(*st["banner"])))
        top = dice.uniform() < 0.2
        y0 = 0 if top else h - bar
        fill = alpha("banner_fill")
        draw.rectangle([0, y0, w, y0 + bar], fill=(20, 20, 20, fill) if light else (245, 245, 245, fill))
        typeface = font(bar * 0.55, face)
        text = f"{word} · {number}" if dice.uniform() < 0.7 else word
        tw, th = _text_size(draw, text, typeface)
        x = int(dice.uniform(0.02, max(0.03, 1 - tw / w - 0.02)) * w)
        draw.text((x, y0 + (bar - th) // 2), text, fill=ink + (alpha("banner_text"),), font=typeface)
    elif layout == "corner":
        typeface = font(h * dice.uniform(*st["corner"]), face)
        text = dice.choice((word, f"© {word}", f"{word} {number}"))
        tw, th = _text_size(draw, text, typeface)
        x = int(w * 0.02) if dice.uniform() < 0.5 else max(0, w - tw - int(w * 0.02))
        y = int(h * 0.02) if dice.uniform() < 0.3 else max(0, h - th - int(h * 0.04))
        if dice.uniform() < 0.4:
            box = st["corner_box"]
            draw.rectangle([x - 3, y - 2, x + tw + 3, y + th + 6],
                           fill=(0, 0, 0, round(255 * box[0])) if light else (255, 255, 255, round(255 * box[1])))
        draw.text((x, y), text, fill=ink + (int(255 * dice.uniform(*st["corner_alpha"])),), font=typeface)
    elif layout == "diagonal":
        typeface = font(w * dice.uniform(*st["diagonal"]), face)
        tw, th = _text_size(draw, word, typeface)
        draw.text(((w - tw) // 2, (h - th) // 2), word, fill=ink + (int(255 * dice.uniform(*st["diagonal_alpha"])),), font=typeface)
        layer = layer.rotate(dice.uniform(-40, 40), center=(w // 2, h // 2))
    elif layout == "id_box":
        typeface = font(h * dice.uniform(*st["id_box"]), 0)  # white digits on a black box, as on archive photos
        text = number[:7]
        tw, th = _text_size(draw, text, typeface)
        x = max(0, w - tw - 10) if dice.uniform() < 0.7 else 4
        y = max(0, h - th - 10)
        draw.rectangle([x - 4, y - 3, x + tw + 4, y + th + 6], fill=(0, 0, 0, alpha("id_fill")))
        draw.text((x, y), text, fill=(255, 255, 255, 255), font=typeface)
    elif layout == "vertical":
        typeface = font(h * dice.uniform(*st["vertical"]), face)
        text = f"{word} - {number}"
        tw, th = _text_size(draw, text, typeface)
        strip = Image.new("RGBA", (tw + 4, th + 6), (0, 0, 0, 0))
        ImageDraw.Draw(strip).text((2, 0), text, fill=ink + (int(255 * dice.uniform(*st["vertical_alpha"])),), font=typeface)
        strip = strip.rotate(90, expand=True)
        x = 2 if dice.uniform() < 0.5 else max(0, w - strip.width - 2)
        layer.alpha_composite(strip, (x, max(0, (h - strip.height) // 2)) if strip.height < h else (x, 0))
    else:
        raise ValueError(layout)
    return _overlay(image, layer)


class RandomWatermark:
    """With probability p (per class: fresh / rotten), stamp a training-style watermark on the photo."""

    def __init__(self, probability_by_label: dict[int, float], style: str = "standard"):
        self.probability_by_label, self.style = probability_by_label, style

    def __call__(self, image: Image.Image, label: int) -> Image.Image:
        if float(torch.rand(1)) < self.probability_by_label.get(label, 0.0):
            return stamp(image, style=self.style)
        return image
