"""experiments/sharper_eye (PREREG.md): checkpoints H and HP from the vegetable model retrained at 320 px.

    python experiments/sharper_eye/build.py H     # 320-px vegetable model + TV-12's CLIP head and new-vegetable head
    python experiments/sharper_eye/build.py HP    # 320-px vegetable model + P1's heads (experiments/dirty_potato)

The seed (42 or 43) with the lower Valid loss is used. The same calibration rules as model.pth: src/calibrate.py with the 192
LVIS crops, the combined temperature on Valid, the heat-map scale (median peak over the 150 Valid rotten photos), tau of the
new-vegetable head by the V5-V7 rule.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "watermark"))
from common import REPO, S, W, choose_tau, clip_features, dev_sets, run, vegetable_logits  # noqa: E402
from model_utils import (  # noqa: E402
    CONDITION_NAMES, clip_head_logits, combined_logits, fit_temperature, forward_with_rot_maps, input_size, load_model,
    make_transform, open_image,
)

LVIS = S / "negx/lvis_veg"
HEADS = {"H": REPO / "model.pth", "HP": W / "L1.pth"}   # L1.pth of experiments/t3_weak_spots = P1


def best_seed() -> Path:
    loss = {s: json.load(open(W / f"H320_s{s}/metrics.json"))["metrics"]["val"]["loss"] for s in (42, 43)}
    run_dir = W / f"H320_s{min(loss, key=loss.get)}"
    print("Valid loss per seed", loss, "->", run_dir.name, flush=True)
    return run_dir


def main():
    tag = sys.argv[1]
    out = W / f"{tag}.pth"
    shutil.copy(best_seed() / "model.pth", out)
    run("src/calibrate.py", "--model", out, "--extra-photos", LVIS, "--extra-name", "LVIS real-scene vegetable crops")
    sets = dev_sets(); F = clip_features(sets); B = vegetable_logits(out, "H320", sets)
    ck = torch.load(out, map_location="cpu", weights_only=True)
    heads = torch.load(HEADS[tag], map_location="cpu", weights_only=True)
    current = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    ck["clip_head"] = copy.deepcopy(heads["clip_head"])
    ck["new_vegetable_head"] = copy.deepcopy(heads["new_vegetable_head"])
    _, y = sets["valid"]
    clip_logits = clip_head_logits(torch.tensor(F["valid"]), ck)
    ck["clip_head"]["temperature"] = fit_temperature(combined_logits(torch.tensor(B["valid"]), clip_logits, ck), torch.tensor(y))
    model, _ = load_model(out, torch.device("cpu")); transform = make_transform(input_size(ck))
    paths = [p for p, c in zip(sets["valid"][0], y) if c % len(CONDITION_NAMES) == 1]
    peaks = []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            logits, maps = forward_with_rot_maps(model, ck, torch.stack([transform(open_image(p)) for p in paths[i:i + 32]]))
            species = logits.argmax(1) // len(CONDITION_NAMES)
            peaks += [float(maps[k, s].max()) for k, s in enumerate(species)]
    ck["rot_map"] = {**current["rot_map"], "scale": float(np.median(peaks))}
    ck["rot_map"].pop("pointing_game", None)
    tau, grid = choose_tau(ck, B, F)
    ck["new_vegetable_head"]["switch_probability"] = tau
    ck["new_vegetable_head"]["switch_rule"] = f"smallest tau moving at most 1 old-vegetable development photo; moved per tau: {grid}"
    ck["version"] = f"candidate {tag} (experiments/sharper_eye)"
    torch.save(ck, out)
    print(tag, "input", input_size(ck), "tau", tau, grid, "combined T", round(ck["clip_head"]["temperature"], 4),
          "vegetable T", round(ck["calibration"]["temperature"], 4), flush=True)


if __name__ == "__main__":
    main()
