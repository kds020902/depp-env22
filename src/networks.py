"""The vegetable model: building, loading, seeding, and the rot maps behind the heatmap."""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights, ResNet18_Weights, efficientnet_b0, resnet18,
)

from images import PREPROCESS, preprocess_for
from labels import CLASS_NAMES, CONDITION_NAMES

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
    if len(names) < 2 or names != CLASS_NAMES[:len(names)]:  # 6- and 10-class checkpoints are prefixes of the 16
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


# ---------- where the model sees rot (heatmap) ----------
HEATMAP_LAYERS = {"resnet18": "layer4", "efficientnet_b0": "features"}


def forward_with_rot_maps(model: nn.Module, checkpoint: dict, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Logits plus a 7x7 "rot map" per photo and vegetable, shape (photos, vegetables, h, w): how much each part of the
    photo pushes that vegetable's "rotten" logit above its "fresh" logit.

    The classifier sits on top of average pooling, so the rotten-minus-fresh logit is exactly the mean over locations of
    (w_rotten - w_fresh) . feature(location); each location's term is its share (class activation map, Zhou et al. 2016).
    Only the positive part (evidence for rotten) is kept. For avg+max pooling only the average half is used.
    Pick the map of the vegetable that was finally predicted.
    """
    arch = checkpoint["architecture"]
    store = {}
    hook = dict(model.named_modules())[HEATMAP_LAYERS[arch]].register_forward_hook(
        lambda module, inputs, output: store.update(features=output))
    try:
        logits = model(images)
    finally:
        hook.remove()
    features = store["features"]
    weight = classifier_layer(model, arch).weight[:, :features.shape[1]]
    weight = weight.reshape(-1, len(CONDITION_NAMES), features.shape[1])
    direction = weight[:, CONDITION_NAMES.index("rotten")] - weight[:, CONDITION_NAMES.index("fresh")]
    return logits, torch.relu(torch.einsum("sc,nchw->nshw", direction, features))


def rot_map_overlay(image: Image.Image, rot_map: torch.Tensor, scale: float, size: int = 480,
                    side_by_side: bool = True) -> Image.Image:
    """The plain photo next to a copy with the rot map laid over it in red (stronger red = stronger rot evidence);
    side_by_side=False returns the coloured copy only.
    The map is divided by a fixed scale (not by its own maximum), so a fresh-looking photo stays almost uncoloured."""
    image = image.copy()
    image.thumbnail((size, size))
    level = torch.nn.functional.interpolate((rot_map / scale).clamp(0, 1)[None, None], size=image.size[::-1],
                                            mode="bilinear", align_corners=False)[0, 0].numpy()[..., None]
    color = np.array([230.0, 20.0, 20.0])
    alpha = 0.7 * level
    overlay = Image.fromarray((np.asarray(image) * (1 - alpha) + color * alpha).astype(np.uint8))
    if not side_by_side:
        return overlay
    canvas = Image.new("RGB", (image.width * 2 + 8, image.height), "white")
    canvas.paste(image, (0, 0))
    canvas.paste(overlay, (image.width + 8, 0))
    return canvas
