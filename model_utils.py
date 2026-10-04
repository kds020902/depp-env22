"""Shared offline-safe model loading, preprocessing and label definitions."""

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
from torchvision.models import ResNet18_Weights, resnet18


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


def set_seed(seed: int = 42, threads: int = 6) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(name)


def make_transform():
    return transforms.Compose([
        transforms.Resize(
            tuple(PREPROCESS["resize"]),
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
        transforms.Normalize(PREPROCESS["mean"], PREPROCESS["std"]),
    ])


def read_image(path: str | Path, transform=None) -> torch.Tensor:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return (transform or make_transform())(image)


def create_model(pretrained: str | Path | None = None) -> tuple[nn.Module, str]:
    """Load ImageNet backbone; only a newly initialized six-way FC is trainable."""
    if pretrained is None:
        local = Path(__file__).resolve().parent.parent / "pretrained" / "resnet18-f37072fd.pth"
        if local.is_file():
            pretrained = local
    if pretrained is not None:
        path = Path(pretrained).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
        model = resnet18(weights=None)
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        source = str(path)
    else:
        # Uses torchvision's cache; an initial download may be required.
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        source = "torchvision.ResNet18_Weights.IMAGENET1K_V1"
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.eval()  # BatchNorm statistics remain frozen during feature extraction.
    return model, source


def load_model(path: str | Path, device: torch.device) -> tuple[nn.Module, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "resnet18":
        raise ValueError("Unsupported architecture in checkpoint.")
    if checkpoint.get("class_names") != CLASS_NAMES:
        raise ValueError("Checkpoint class order does not match this application.")
    if checkpoint.get("preprocess") != PREPROCESS:
        raise ValueError("Checkpoint preprocessing does not match this application.")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def state_checksum(state: dict[str, torch.Tensor], exclude_fc: bool = False) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        if exclude_fc and name.startswith("fc."):
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
