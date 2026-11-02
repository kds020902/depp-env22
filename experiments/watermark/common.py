"""Shared parts of the watermark round: development photo sets, cached model outputs, the full 8-vegetable decision.

The photo lists point into the working folder of this project (photos are not in the repository for licence reasons),
so these scripts are a record of what was run.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_utils import (  # noqa: E402
    CLASS_NAMES, clip_class_logits, clip_head_logits, combined_logits, combined_probabilities, input_size,
    is_unknown_combined, load_model, make_transform, new_vegetable_decision, open_image,
)
from stamps import STAMPS  # noqa: E402

S = Path(os.environ["DEPP_WORK"])
W = S / "wm"
TAU_GRID = (0.5, 0.7, 0.8, 0.9, 0.95)
OLD_DEV = ("valid", "odev", "o160", "asian37", "lvis")


def dev_sets() -> dict[str, tuple[list[str], np.ndarray | None]]:
    """name -> (photo paths, class indices or None for photos that should be turned away / have no label)."""
    sets = json.load(open(S / "clip/sets.json"))
    out = {}
    for name, key in (("valid", "valid"), ("odev", "odev"), ("o160", "o160"), ("asian37", "asian37")):
        out[name] = ([p for p, _ in sets[key]], np.array([CLASS_NAMES.index(c) for _, c in sets[key]]))
    for name, key in (("fruit374", "fruit374"), ("coco400", "coco400"), ("lvis", "lvis_veg")):
        out[name] = ([p for p, _ in sets[key]], None)
    pool = [r for r in json.load(open(S / "v7/pool.json")) if r["split"] == "val"]
    out["newval"] = ([r["path"] for r in pool], np.array([CLASS_NAMES.index(r["class_name"]) for r in pool]))
    return out


def valid_watermarks() -> np.ndarray:
    """1 where the Valid photo (in dev_sets()['valid'] order) carries a real watermark."""
    rows = {r["image_path"]: int(r["watermark"]) for r in csv.DictReader(open(REPO / "experiments/watermark/valid_test_watermarks.csv"))}
    paths, _ = dev_sets()["valid"]
    return np.array([rows[str(Path(p).relative_to(REPO / "dataset"))] for p in paths])


def _images(paths: list[str], stamp: str | None):
    for p in paths:
        image = open_image(p)
        yield STAMPS[stamp](image) if stamp else image


def clip_features(sets, cache: Path = W / "clip_cache.npz") -> dict[str, np.ndarray]:
    """CLIP image features (the same model as the CLIP heads); they do not depend on the candidate, so computed once."""
    from clip_model import ClipModel
    have = dict(np.load(cache)) if cache.exists() else {}
    clip = None
    for name, stamp in [(n, None) for n in sets] + [("valid_S1", "S1"), ("valid_S2", "S2")]:
        if name in have:
            continue
        clip = clip or ClipModel("cpu")
        paths = sets["valid" if stamp else name][0]
        images = list(_images(paths, stamp))
        with torch.inference_mode():
            have[name] = torch.cat([clip.image_features(images[i:i + 32]).cpu() for i in range(0, len(images), 32)]).numpy()
        np.savez(cache, **have)
    return have


def vegetable_logits(model_path: Path, tag: str, sets) -> dict[str, np.ndarray]:
    cache = W / f"b_{tag}.npz"
    have = dict(np.load(cache)) if cache.exists() else {}
    model, checkpoint = load_model(model_path, torch.device("cpu")); transform = make_transform(input_size(checkpoint))
    for name, stamp in [(n, None) for n in sets] + [("valid_S1", "S1"), ("valid_S2", "S2")]:
        if name in have:
            continue
        paths = sets["valid" if stamp else name][0]
        out = []
        with torch.inference_mode():
            batch = []
            for image in _images(paths, stamp):
                batch.append(transform(image))
                if len(batch) == 32:
                    out.append(model(torch.stack(batch))); batch = []
            if batch:
                out.append(model(torch.stack(batch)))
        have[name] = torch.cat(out).numpy()
        np.savez(cache, **have)
    return have


def system(checkpoint: dict, b: np.ndarray, f: np.ndarray):
    """Five-vegetable probabilities and unknown flags, then the full 8-vegetable answer (chosen class, turned away)."""
    logits, feats = torch.tensor(b), torch.tensor(f)
    clip_logits = clip_head_logits(feats, checkpoint)
    probabilities = combined_probabilities(combined_logits(logits, clip_logits, checkpoint), checkpoint)
    unknown = is_unknown_combined(logits, clip_logits, checkpoint, feats)
    _, index, still_unknown, switched = new_vegetable_decision(probabilities, unknown, feats, checkpoint)
    return {"five": probabilities.numpy(), "five_unknown": unknown.numpy(), "index": index.numpy(),
            "unknown": still_unknown.numpy(), "switched": switched.numpy(),
            "vegetable_model": (logits / checkpoint["calibration"]["temperature"]).softmax(1).numpy(),
            "clip_head": clip_class_logits(clip_logits, checkpoint).softmax(1).numpy()}


def kept_right_per_class(checkpoint: dict, names=("valid", "odev", "o160", "asian37")) -> dict:
    """set -> class -> "kept and right / photos" on the labelled development sets (TV-12's vegetable-model outputs, b_WC)."""
    sets = dev_sets(); F = clip_features(sets); B = dict(np.load(W / "b_WC.npz")); out = {}
    for name in names:
        r = system(checkpoint, B[name], F[name]); y = sets[name][1]
        out[name] = {CLASS_NAMES[c]: f"{int(((r['index'] == c) & ~r['unknown'])[y == c].sum())}/{int((y == c).sum())}"
                     for c in range(10) if (y == c).any()}
    return out


def choose_tau(checkpoint: dict, B: dict, F: dict) -> tuple[float, dict]:
    """The V5/V6/V7 rule: the smallest tau that moves at most one old-vegetable development photo, which the
    five-vegetable system answered, to a new vegetable."""
    head = checkpoint["new_vegetable_head"]; keep = head["switch_probability"]; grid = {}
    for tau in TAU_GRID:
        head["switch_probability"] = tau
        grid[tau] = sum(int((~r["five_unknown"] & r["switched"]).sum()) for r in (system(checkpoint, B[n], F[n]) for n in OLD_DEV))
    head["switch_probability"] = keep
    return min([t for t, m in grid.items() if m <= 1] or [TAU_GRID[-1]]), grid


if __name__ == "__main__":
    os.makedirs(W, exist_ok=True)
    sets = dev_sets()
    print({k: len(v[0]) for k, v in sets.items()})
    F = clip_features(sets)
    print({k: v.shape for k, v in F.items()})
