"""Shared model loading, preprocessing and label definitions."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights, ResNet18_Weights, efficientnet_b0, resnet18,
)


# Class id = 2 x species + condition. New species are appended, so older 6-class checkpoints keep their ids.
CLASS_NAMES = [
    "cucumber_fresh", "cucumber_rotten", "potato_fresh",
    "potato_rotten", "tomato_fresh", "tomato_rotten",
    "bellpepper_fresh", "bellpepper_rotten", "carrot_fresh", "carrot_rotten",
]
SPECIES_NAMES = ["cucumber", "potato", "tomato", "bellpepper", "carrot"]
CONDITION_NAMES = ["fresh", "rotten"]
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
# name -> (builder, ImageNet weights, official weight file name, classifier attribute path)
ARCHITECTURES = {
    "resnet18": (resnet18, ResNet18_Weights.IMAGENET1K_V1, "resnet18-f37072fd.pth", ("fc",)),
    "efficientnet_b0": (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1,
                        "efficientnet_b0_rwightman-7f5810bc.pth", ("classifier", "1")),
}
PRETRAINED_DIR = Path(__file__).resolve().parent.parent / "pretrained"  # <repo>/pretrained


def set_seed(seed: int = 42, threads: int = 6) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    # warn_only: a few CUDA backward kernels have no deterministic version.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(name)


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


def classifier_layer(model: nn.Module, arch: str) -> nn.Linear:
    module = model
    for name in ARCHITECTURES[arch][3]:
        module = module[int(name)] if name.isdigit() else getattr(module, name)
    return module


def set_classifier_layer(model: nn.Module, arch: str, layer: nn.Module) -> None:
    *parents, last = ARCHITECTURES[arch][3]
    module = model
    for name in parents:
        module = module[int(name)] if name.isdigit() else getattr(module, name)
    if last.isdigit():
        module[int(last)] = layer
    else:
        setattr(module, last, layer)


class AvgMaxPool(nn.Module):
    """Average and max pooling side by side, so one small suspicious patch is not averaged away."""

    def forward(self, x):
        return torch.cat([x.mean((2, 3), keepdim=True), x.amax((2, 3), keepdim=True)], 1)


def _apply_pooling(model: nn.Module, arch: str, pooling: str) -> None:
    """Swap the global pooling before the classifier ("avg" = torchvision default)."""
    if pooling == "avg":
        return
    if pooling != "avgmax":
        raise ValueError(f"Unknown pooling {pooling!r}; choose 'avg' or 'avgmax'")
    model.avgpool = AvgMaxPool()
    old = classifier_layer(model, arch)
    set_classifier_layer(model, arch, nn.Linear(old.in_features * 2, old.out_features))


def create_model(arch: str = "resnet18", pretrained: str | Path | None = None,
                 pooling: str = "avg", num_classes: int = len(CLASS_NAMES)) -> tuple[nn.Module, str]:
    """ImageNet backbone with a newly initialised classifier (one output per class)."""
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}; choose from {sorted(ARCHITECTURES)}")
    builder, weights, file_name, _ = ARCHITECTURES[arch]
    if pretrained is None and (PRETRAINED_DIR / file_name).is_file():
        pretrained = PRETRAINED_DIR / file_name
    if pretrained is not None:
        path = Path(pretrained).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
        model = builder(weights=None)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
        source = str(path)
    else:
        # Uses torchvision's cache; the first run downloads the official weights.
        model = builder(weights=weights)
        source = f"torchvision.{type(weights).__name__}.{weights.name}"
    old = classifier_layer(model, arch)
    set_classifier_layer(model, arch, nn.Linear(old.in_features, num_classes))
    _apply_pooling(model, arch, pooling)
    return model, source


def load_model(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    arch = checkpoint.get("architecture")
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture in checkpoint: {arch!r}")
    names = checkpoint.get("class_names") or []
    if len(names) < 2 or names != CLASS_NAMES[:len(names)]:  # a 6-class checkpoint is a prefix of the 10
        raise ValueError("Checkpoint class order does not match this application.")
    if checkpoint.get("preprocess") != preprocess_for(input_size(checkpoint)):
        raise ValueError("Checkpoint preprocessing does not match this application.")
    model = ARCHITECTURES[arch][0](weights=None)
    old = classifier_layer(model, arch)
    set_classifier_layer(model, arch, nn.Linear(old.in_features, len(names)))
    _apply_pooling(model, arch, checkpoint.get("pooling", "avg"))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def input_size(checkpoint: dict) -> int:
    """Square input size a checkpoint was trained with (224 for older checkpoints)."""
    return int(checkpoint.get("preprocess", PREPROCESS)["resize"][0])


def state_checksum(state: dict[str, torch.Tensor], exclude_prefixes: tuple[str, ...] = ()) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        if name.startswith(exclude_prefixes):
            continue
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def marginal_probabilities(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum the joint (species x condition) probabilities, rather than reusing joint argmax labels."""
    joint = probabilities.reshape(-1, probabilities.shape[1] // len(CONDITION_NAMES), len(CONDITION_NAMES))
    return joint.sum(dim=2), joint.sum(dim=1)


# ---------- calibrated probability and the "not a trained vegetable" check ----------
UNKNOWN_MESSAGE = "학습된 데이터가 아닙니다"


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature scaling: one number T so that softmax(logits / T) matches how often the model is right
    (fitted on the validation split by minimising cross-entropy)."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_t.detach().exp())


def unknown_score(logits: torch.Tensor) -> torch.Tensor:
    """Energy score, -logsumexp(logits): high when no trained class fits the photo well."""
    return -torch.logsumexp(logits, dim=1)


def fit_unknown_threshold(val_logits: torch.Tensor, accept_rate: float = 0.95) -> float:
    """Threshold that still accepts accept_rate of the validation photos (all real vegetables)."""
    return float(torch.quantile(unknown_score(val_logits), accept_rate))


def calibrated_probabilities(logits: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    temperature = checkpoint.get("calibration", {}).get("temperature", 1.0)
    return (logits / temperature).softmax(1)


def is_unknown(logits: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    threshold = checkpoint.get("unknown_check", {}).get("threshold")
    if threshold is None:  # older checkpoints have no check
        return torch.zeros(len(logits), dtype=torch.bool)
    return unknown_score(logits) > threshold


def calibration_extras(val_logits: torch.Tensor, val_labels: torch.Tensor, accept_rate: float = 0.95) -> dict:
    """The checkpoint entries for calibrated probabilities and the unknown-photo check, fitted on Valid only."""
    return {
        "calibration": {"method": "temperature scaling", "temperature": fit_temperature(val_logits, val_labels),
                        "fitted_on": "val"},
        "unknown_check": {"method": "energy score -logsumexp(logits)",
                          "threshold": fit_unknown_threshold(val_logits, accept_rate),
                          "valid_accept_rate": accept_rate, "fitted_on": "val", "message": UNKNOWN_MESSAGE},
    }
