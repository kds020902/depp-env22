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


CLASS_NAMES = [
    "cucumber_fresh", "cucumber_rotten", "potato_fresh",
    "potato_rotten", "tomato_fresh", "tomato_rotten",
]
SPECIES_NAMES = ["cucumber", "potato", "tomato"]
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
PRETRAINED_DIR = Path(__file__).resolve().parent / "pretrained"


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


def make_transform():
    """Evaluation/prediction transform: the whole photo squeezed to 224x224."""
    return transforms.Compose([
        transforms.Resize(
            tuple(PREPROCESS["resize"]),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(PREPROCESS["mean"], PREPROCESS["std"]),
    ])


def make_train_transform():
    """Training augmentation: random crop/scale, flips, small rotation and colour change."""
    return transforms.Compose([
        transforms.RandomResizedCrop(
            tuple(PREPROCESS["resize"]), scale=(0.6, 1.0), ratio=(3 / 4, 4 / 3), antialias=True,
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


def create_model(arch: str = "resnet18", pretrained: str | Path | None = None) -> tuple[nn.Module, str]:
    """ImageNet backbone with a newly initialised six-way classifier."""
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
    set_classifier_layer(model, arch, nn.Linear(old.in_features, len(CLASS_NAMES)))
    return model, source


def load_model(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    arch = checkpoint.get("architecture")
    if arch not in ARCHITECTURES:
        raise ValueError(f"Unsupported architecture in checkpoint: {arch!r}")
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise ValueError("Checkpoint class order does not match this application.")
    if checkpoint.get("preprocess") != PREPROCESS:
        raise ValueError("Checkpoint preprocessing does not match this application.")
    model = ARCHITECTURES[arch][0](weights=None)
    old = classifier_layer(model, arch)
    set_classifier_layer(model, arch, nn.Linear(old.in_features, len(CLASS_NAMES)))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


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
    """Sum the six joint probabilities, rather than reusing joint argmax labels."""
    joint = probabilities.reshape(-1, len(SPECIES_NAMES), len(CONDITION_NAMES))
    return joint.sum(dim=2), joint.sum(dim=1)
