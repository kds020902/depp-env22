"""Model inputs of the photos used in addenda 2-4 (b = vegetable-model logits, X = CLIP image features), so the runs repeat without photos.

    python experiments/staged_ttv1/features.py PHOTOS.csv results3.json OUT.npz

Only the photos kept in results3.json (not near a blind-test or dataset photo) are stored.
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from images import open_image  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402
from predict import Classifier  # noqa: E402

photos, kept, out = sys.argv[1:]
kept = {p["id"] for p in json.load(open(kept))["photos"]}
system = Classifier(REPO / "model.pth", torch.device("cpu"))
ids, b, X, y, group = [], [], [], [], []
for r in csv.DictReader(open(photos, encoding="utf-8")):
    if r["id"] not in kept:
        continue
    image = open_image(r["path"])
    with torch.inference_mode():
        b.append(system.model(system.transform(image).unsqueeze(0))[0].numpy())
    X.append(system.clip.image_features([image])[0].numpy())
    ids.append(r["id"]); y.append(CLASS_NAMES.index(r["class_name"])); group.append(r["group"])
np.savez_compressed(out, ids=np.array(ids), b=np.array(b, np.float32), X=np.array(X, np.float32), y=np.array(y, np.int16), group=np.array(group))
print(len(ids))
