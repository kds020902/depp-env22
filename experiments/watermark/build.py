"""Build the candidate checkpoints of the watermark round (PREREG.md).

    python experiments/watermark/build.py WA          # CLIP head with watermarked copies, vegetable model unchanged
    python experiments/watermark/build.py W0 W0_s42   # control: retrained vegetable model (no watermarks) + current CLIP head
    python experiments/watermark/build.py WB          # retrained vegetable model with watermarks (seed by Valid loss) + WA's head

Every candidate gets the same calibration rules as model.pth: the vegetable model's temperature and energy threshold
(src/calibrate.py with the 192 LVIS crops), the combined temperature on Valid, the heat-map scale (median peak over the
150 Valid rotten photos), and tau of the new-vegetable head by the V5-V7 rule. The new-vegetable head is otherwise kept.
"""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys

import numpy as np
import torch

from common import REPO, S, W, choose_tau, clip_features, dev_sets, vegetable_logits
from model_utils import (
    CONDITION_NAMES, clip_head_logits, combined_logits, fit_temperature, forward_with_rot_maps, input_size, load_model,
    make_transform, open_image,
)

LVIS = S / "negx/lvis_veg"
EXTRA = S / "negx/extra_train_N3.csv"
WATERMARK_HEAD = ["--watermark-fresh", "1.0", "--watermark-rotten", "0.25", "--watermark-other", "0.5"]


def best_seed(name: str) -> Path:
    """The seed with the lower Valid cross-entropy (no watermark result is used)."""
    loss = {s: json.load(open(W / f"{name}_s{s}/metrics.json"))["metrics"]["val"]["loss"] for s in (42, 43)}
    run_dir = W / f"{name}_s{min(loss, key=loss.get)}"
    print("Valid loss per seed", loss, "->", run_dir.name, flush=True)
    return run_dir


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=REPO)


def finish(path, tag: str, head_source=None):
    """Combined temperature (if the CLIP head is taken from another checkpoint), heat-map scale, tau."""
    sets = dev_sets(); F = clip_features(sets); B = vegetable_logits(path, tag, sets)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    current = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    _, y = sets["valid"]
    if head_source is not None:
        checkpoint["clip_head"] = copy.deepcopy(head_source["clip_head"])
        clip_logits = clip_head_logits(torch.tensor(F["valid"]), checkpoint)
        checkpoint["clip_head"]["temperature"] = fit_temperature(
            combined_logits(torch.tensor(B["valid"]), clip_logits, checkpoint), torch.tensor(y))
    if "rot_map" not in checkpoint or head_source is not None:
        model, _ = load_model(path, torch.device("cpu")); transform = make_transform(input_size(checkpoint))
        paths = [p for p, c in zip(sets["valid"][0], y) if c % len(CONDITION_NAMES) == 1]
        peaks = []
        with torch.inference_mode():
            for i in range(0, len(paths), 32):
                logits, maps = forward_with_rot_maps(model, checkpoint, torch.stack([transform(open_image(p)) for p in paths[i:i + 32]]))
                species = logits.argmax(1) // len(CONDITION_NAMES)
                peaks += [float(maps[k, s].max()) for k, s in enumerate(species)]
        checkpoint["rot_map"] = {**current["rot_map"], "scale": float(np.median(peaks))}
        checkpoint["rot_map"].pop("pointing_game", None)
    checkpoint["new_vegetable_head"] = copy.deepcopy(current["new_vegetable_head"])
    tau, grid = choose_tau(checkpoint, B, F)
    checkpoint["new_vegetable_head"]["switch_probability"] = tau
    checkpoint["new_vegetable_head"]["switch_rule"] = f"smallest tau moving at most 1 old-vegetable development photo; moved per tau: {grid}"
    torch.save(checkpoint, path)
    print(tag, "tau", tau, grid, "combined T", round(checkpoint["clip_head"]["temperature"], 4), flush=True)


def main():
    tag = sys.argv[1]
    out = W / f"{tag}.pth"
    if tag == "V7":
        shutil.copy(REPO / "model.pth", out)
        finish(out, tag)  # must give tau 0.8 again (pipeline check)
    elif tag in ("WA", "WAL"):  # WAL (round 2): the same with light watermarks
        shutil.copy(REPO / "model.pth", out)
        run("src/train_clip_head.py", "--model", out, "--extra-photos", LVIS, "--extra-train", EXTRA, "--clip-check-only",
            *WATERMARK_HEAD, "--feature-cache", W / "head_features.npz",
            *(["--watermark-style", "light"] if tag == "WAL" else []))
        finish(out, tag)
    else:
        if tag == "WB":
            run_dir = best_seed("WB")
            head_source = torch.load(W / "WA.pth", map_location="cpu", weights_only=True)
        elif tag in ("WC", "WD"):  # round 2: light-watermark vegetable model + light-watermark head (WC) or V7's head (WD)
            run_dir = best_seed("WC")
            head_source = torch.load(W / "WAL.pth" if tag == "WC" else REPO / "model.pth", map_location="cpu", weights_only=True)
        else:
            run_dir = W / sys.argv[2]
            head_source = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
        shutil.copy(run_dir / "model.pth", out)
        run("src/calibrate.py", "--model", out, "--extra-photos", LVIS, "--extra-name", "LVIS real-scene vegetable crops")
        finish(out, tag, head_source)


if __name__ == "__main__":
    main()
