"""PC: a normal-only patch memory for sweet potatoes, after PatchCore (PREREG.md, addendum 4).

    python experiments/staged_ttv1/run5.py PHOTOS.csv results3.json OUT.json [--maps FOLDER]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from images import make_transform, open_image  # noqa: E402
from networks import load_model  # noqa: E402

FOLDS, GRID = 5, 28


def patch_features(model, x):
    """784 patch features of one photo: blocks 3 (28 x 28) and 5 (14 x 14, upsampled), 3 x 3 averaged, L2-normalised."""
    maps = []
    for i, block in enumerate(model.features):
        x = block(x)
        if i in (3, 5):
            maps.append(torch.nn.functional.interpolate(x, size=(GRID, GRID), mode="bilinear", align_corners=False))
        if i == 5:
            break
    f = torch.nn.functional.avg_pool2d(torch.cat(maps, 1), 3, 1, 1)[0].flatten(1).T
    return torch.nn.functional.normalize(f, dim=1)


def distances(f, bank):
    return torch.cdist(f, bank).min(1).values.reshape(GRID, GRID)


def photo_score(d, centre=False):
    if centre:
        e = round(GRID * 0.2)
        d = d[e:GRID - e, e:GRID - e]
    v = d.flatten()
    return float(v.topk(max(1, len(v) // 100)).values.mean())


def save_map(image, d, scale, path, title):
    from PIL import Image, ImageDraw
    image = image.copy(); image.thumbnail((320, 320))
    w, h = image.size
    level = np.clip((d.numpy() - scale[0]) / (scale[1] - scale[0]), 0, 1)
    alpha = np.asarray(Image.fromarray((level * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), float)[..., None] / 255 * 0.7
    overlay = Image.fromarray((np.asarray(image) * (1 - alpha) + np.array([230.0, 20.0, 20.0]) * alpha).astype(np.uint8))
    canvas = Image.new("RGB", (w * 2 + 8, h + 20), "white")
    canvas.paste(image, (0, 20)); canvas.paste(overlay, (w + 8, 20))
    ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    canvas.save(path, quality=88)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photos", type=Path); parser.add_argument("kept", type=Path); parser.add_argument("out", type=Path)
    parser.add_argument("--maps", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_model(REPO / "model.pth", torch.device("cpu"))
    transform = make_transform(224)
    kept = {p["id"]: p for p in json.load(open(args.kept))["photos"]}
    rows = [r for r in csv.DictReader(open(args.photos, encoding="utf-8")) if r["id"] in kept and r["class_name"].startswith("sweetpotato")]
    images = {r["id"]: open_image(r["path"]) for r in rows}
    with torch.inference_mode():
        feats = {r["id"]: patch_features(model, transform(images[r["id"]]).unsqueeze(0)) for r in rows}
    fresh = [r["id"] for r in rows if r["class_name"] == "sweetpotato_fresh"]
    rotten = [r["id"] for r in rows if r["class_name"] == "sweetpotato_rotten"]
    maps = {}
    sums = {i: torch.zeros(GRID, GRID) for i in rotten}
    for fold in range(FOLDS):
        bank = torch.cat([feats[i] for k, i in enumerate(fresh) if k % FOLDS != fold])
        for k, i in enumerate(fresh):
            if k % FOLDS == fold:
                maps[i] = distances(feats[i], bank)
        for i in rotten:
            sums[i] += distances(feats[i], bank) / FOLDS
    maps.update(sums)
    result = {"photos": [], "auroc": {}, "caught_at_90_percent_of_fresh_passing": {}}
    for r in rows:
        result["photos"].append({"id": r["id"], "class_name": r["class_name"], "group": r["group"], "tv13": kept[r["id"]]["tv13"],
                                 "score": round(photo_score(maps[r["id"]]), 4), "score_centre": round(photo_score(maps[r["id"]], True), 4)})
    P = result["photos"]
    for key in ("score", "score_centre"):
        f = [p[key] for p in P if p["class_name"] == "sweetpotato_fresh"]
        cut = float(np.quantile(f, 0.9))
        for name, pick in (("all rotten", lambda p: True), ("same source (root_and_tomato = iNaturalist)", lambda p: p["group"] == "root_and_tomato"),
                           ("sweetpotato_photos train", lambda p: p["group"] == "sweetpotato_photos train"), ("user photos 2", lambda p: p["group"] == "user photos 2")):
            r_ = [p[key] for p in P if p["class_name"] == "sweetpotato_rotten" and pick(p)]
            result["auroc"][f"{key} / {name}"] = round(float(roc_auc_score([0] * len(f) + [1] * len(r_), f + r_)), 3)
            result["caught_at_90_percent_of_fresh_passing"][f"{key} / {name}"] = [int(sum(v > cut for v in r_)), len(r_)]
    json.dump(result, open(args.out, "w"), indent=1, ensure_ascii=False)
    for k, v in result["auroc"].items():
        print(k, v, result["caught_at_90_percent_of_fresh_passing"][k])
    if args.maps:
        args.maps.mkdir(parents=True, exist_ok=True)
        allv = torch.cat([maps[i].flatten() for i in fresh])
        scale = (float(allv.quantile(0.5)), float(allv.quantile(0.999)))
        for p in P:
            name = f"{p['class_name']}__{p['group'].split()[0]}__{Path(p['id']).stem}".replace("/", "_")
            save_map(images[p["id"]], maps[p["id"]], scale, args.maps / (name + ".jpg"), f"{p['class_name']} score {p['score']:.3f}")


if __name__ == "__main__":
    main()
