"""experiments/sweetpotato_prompts: store the model inputs of the potato / sweet potato development photos.

    DEPP_WORK=... python experiments/sweetpotato_prompts/dev_features.py

The photos are the potato and sweet potato rows of experiments/root_and_tomato/new_train.csv that passed its dedup
(web photos collected for R1-R3; TV-12 was never trained on them and none is in a blind test set). They stay in the working
folder ($DEPP_WORK/rt/img, licences); dev_features.npz holds what the decision code needs, so the comparison runs without them:
X = CLIP image features (ViT-B-16 DataComp-XL, L2-normalised), b = the vegetable model's logits (model.pth), y, ids.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from clip_model import ClipModel  # noqa: E402
from images import make_transform, open_image  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402
from networks import input_size, load_model  # noqa: E402

SOURCE = REPO / "experiments/root_and_tomato"


def main():
    photos = Path(os.environ["DEPP_WORK"]) / "rt/img"
    left_out = set(json.load(open(SOURCE / "dedup.json"))["left_out"])
    rows = [r for r in csv.DictReader(open(SOURCE / "new_train.csv", encoding="utf-8"))
            if r["class_name"].startswith(("potato_", "sweetpotato_")) and r["id"] not in left_out]
    model, checkpoint = load_model(REPO / "model.pth", torch.device("cpu"))
    transform, clip = make_transform(input_size(checkpoint)), ClipModel("cpu")
    logits, features = [], []
    with torch.inference_mode():
        for i in range(0, len(rows), 32):
            images = [open_image(photos / f"{r['id']}.jpg") for r in rows[i:i + 32]]
            logits.append(model(torch.stack([transform(image) for image in images])))
            features.append(clip.image_features(images).cpu())
    np.savez_compressed(HERE / "dev_features.npz", X=torch.cat(features).numpy().astype(np.float32),
                        b=torch.cat(logits).numpy().astype(np.float32),
                        y=np.array([CLASS_NAMES.index(r["class_name"]) for r in rows], np.int16),
                        ids=np.array([r["id"] for r in rows]))
    print({c: sum(r["class_name"] == c for r in rows) for c in CLASS_NAMES if any(r["class_name"] == c for r in rows)})


if __name__ == "__main__":
    main()
