"""Opening photos and the preprocessing / augmentation transforms."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps
import torch
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PREPROCESS = {
    "exif_transpose": True,
    "color_mode": "RGB",
    "resize": [224, 224],
    "interpolation": "bilinear",
    "antialias": True,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}


def preprocess_for(size: int = 224) -> dict:
    """PREPROCESS with a different square input size (224 = the original setting)."""
    return {**PREPROCESS, "resize": [size, size]}


def make_transform(size: int = 224):
    """Evaluation/prediction transform: the whole photo squeezed to size x size."""
    return transforms.Compose([
        transforms.Resize(
            (size, size),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(PREPROCESS["mean"], PREPROCESS["std"]),
    ])


class RandomLowRes:
    """With probability p, shrink the photo so its short side is low..high px, like a small web thumbnail.
    The random crop that follows enlarges it again, so the model also learns from blurry, small photos."""

    def __init__(self, p: float, low: int = 48, high: int = 112):
        self.p, self.low, self.high = p, low, high

    def __call__(self, image: Image.Image) -> Image.Image:
        if float(torch.rand(1)) >= self.p:
            return image
        scale = int(torch.randint(self.low, self.high + 1, (1,))) / min(image.size)
        if scale >= 1:
            return image
        return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                            Image.BILINEAR)


def make_train_transform(size: int = 224, crop_scale_min: float = 0.6, lowres_prob: float = 0.0):
    """Training augmentation: random crop/scale, flips, small rotation and colour change
    (optionally first a random shrink to thumbnail size)."""
    lowres = [RandomLowRes(lowres_prob)] if lowres_prob > 0 else []
    return transforms.Compose(lowres + [
        transforms.RandomResizedCrop(
            (size, size), scale=(crop_scale_min, 1.0), ratio=(3 / 4, 4 / 3), antialias=True,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15, fill=(124, 116, 104)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(PREPROCESS["mean"], PREPROCESS["std"]),
    ])


def open_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def read_image(path: str | Path, transform=None) -> torch.Tensor:
    return (transform or make_transform())(open_image(path))
