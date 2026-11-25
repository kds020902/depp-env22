"""SPE: rot evidence for sweet potatoes from nine crops, with what TV-13 already holds (PREREG.md, addendum 2).

    python experiments/staged_ttv1/run3.py PHOTOS.csv OUT.json [--bank FEATURES.npz] [--maps FOLDER]

PHOTOS.csv: columns id,path,class_name,group. Photos are not in the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from decision import new_vegetable_logits  # noqa: E402
from images import open_image  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402
from predict import Classifier  # noqa: E402

FRESH, ROTTEN = CLASS_NAMES.index("sweetpotato_fresh"), CLASS_NAMES.index("sweetpotato_rotten")
CUTS = (-2, -1, 0, 1, 2, 3, 4)


def crops(image):
    """Nine crops of half the width and height (3 x 3, half overlapping), row by row."""
    w, h = image.size
    return [image.crop((x * w // 4, y * h // 4, x * w // 4 + w // 2, y * h // 4 + h // 2)) for y in range(3) for x in range(3)]


def evidence(system: Classifier, image) -> list[float]:
    z = new_vegetable_logits(system.clip.image_features(crops(image)).cpu(), system.checkpoint["new_vegetable_head"])
    return (z[:, ROTTEN] - z[:, FRESH]).tolist()


def answer_with(base: str, cells: list[float], cut: float, second: bool = False) -> str:
    top = sorted(cells)[-2 if second else -1]
    return "sweetpotato_rotten" if base == "sweetpotato_fresh" and top > cut else base


def save_map(image, cells, path: Path):
    from PIL import Image, ImageDraw
    image = image.copy()
    image.thumbnail((360, 360))
    w, h = image.size
    heat = np.zeros((4, 4)); count = np.zeros((4, 4))
    for i, v in enumerate(cells):     # each crop covers 2 x 2 of the 4 x 4 quarter cells
        y, x = divmod(i, 3)
        heat[y:y + 2, x:x + 2] += max(v, 0); count[y:y + 2, x:x + 2] += 1
    level = np.clip(heat / count / 4, 0, 1)
    alpha = np.asarray(Image.fromarray((level * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), float)[..., None] / 255 * 0.7
    overlay = Image.fromarray((np.asarray(image) * (1 - alpha) + np.array([230.0, 20.0, 20.0]) * alpha).astype(np.uint8))
    canvas = Image.new("RGB", (w * 2 + 8, h + 22), "white")
    canvas.paste(image, (0, 22)); canvas.paste(overlay, (w + 8, 22))
    ImageDraw.Draw(canvas).text((4, 4), f"{path.stem}  max {max(cells):+.1f}", fill="black")
    canvas.save(path, quality=88)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--maps", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    system = Classifier(REPO / "model.pth", torch.device("cpu"))
    bank = torch.tensor(np.load(args.bank, allow_pickle=True)["X"]).float() if args.bank else None
    rows, left_out = [], []
    for r in csv.DictReader(open(args.photos, encoding="utf-8")):
        image = open_image(r["path"])
        if bank is not None:
            near = float((system.clip.image_features([image]).cpu().float() @ bank.T).max())
            if near >= 0.95:
                left_out.append({"id": r["id"], "cosine": round(near, 3)})
                continue
        a = system.classify(image, 0.9)
        base = "turned_away" if a["unknown"] else CLASS_NAMES[a["index"]]
        cells = evidence(system, image)
        rows.append({"id": r["id"], "class_name": r["class_name"], "group": r["group"], "tv13": base,
                     "probability": round(a["probability"], 4), "held": a["held"], "cells": [round(v, 3) for v in cells]})
        if args.maps and r["class_name"].startswith("sweetpotato"):
            args.maps.mkdir(parents=True, exist_ok=True)
            save_map(image, cells, args.maps / (f"{r['class_name']}__{Path(r['id']).stem}".replace('/', '_') + '.jpg'))
    groups = {}
    for r in rows:
        g = groups.setdefault(f"{r['group']} / {r['class_name']}", {"n": 0, "TV-13": 0, "SPE": 0, "SPE2": 0, "answers_changed_SPE": 0, "tv13_answers": {}})
        g["n"] += 1
        g["tv13_answers"][r["tv13"]] = g["tv13_answers"].get(r["tv13"], 0) + 1
        for name, ans in (("TV-13", r["tv13"]), ("SPE", answer_with(r["tv13"], r["cells"], 0)), ("SPE2", answer_with(r["tv13"], r["cells"], 0, True))):
            g[name] += ans == r["class_name"]
        g["answers_changed_SPE"] += answer_with(r["tv13"], r["cells"], 0) != r["tv13"]
    curve = {}
    for cut in CUTS:
        flipped = [r for r in rows if answer_with(r["tv13"], r["cells"], cut) != r["tv13"]]
        curve[str(cut)] = {"rotten_sweet_potato_caught": sum(r["class_name"] == "sweetpotato_rotten" for r in flipped),
                           "fresh_sweet_potato_lost": sum(r["class_name"] == "sweetpotato_fresh" for r in flipped),
                           "other_photos_changed": sum(not r["class_name"].startswith("sweetpotato") for r in flipped)}
    json.dump({"left_out_near_blind_or_dataset": left_out, "groups": groups, "curve_by_cut": curve, "photos": rows},
              open(args.out, "w"), indent=1, ensure_ascii=False)
    for k, g in sorted(groups.items()):
        print(k, g)
    print("left out", left_out)
    for k, v in curve.items():
        print(k, v)


if __name__ == "__main__":
    main()
