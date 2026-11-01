"""V7's Test answers split by session overlap (README.md). Record script: uses the cached Test features of the working folder.

    python experiments/session_overlap/score.py
"""
import collections, csv, json, sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "watermark"))
from common import REPO, S, CLASS_NAMES, system  # noqa: E402

rows = [r for r in csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")) if r["split"] == "test"]
y = np.array([CLASS_NAMES.index(r["class_name"]) for r in rows])
same_paths = {r["test_image"] for r in csv.DictReader(open(HERE / "same_session.csv"))}
same = np.array([r["image_path"] in same_paths for r in rows])
wmap = {r["image_path"]: int(r["watermark"]) for r in csv.DictReader(open(REPO / "experiments/watermark/valid_test_watermarks.csv"))}
wm = np.array([wmap[r["image_path"]] for r in rows]).astype(bool)
T = np.load(S / "clipcls/test_features.npz")  # vegetable-model logits and CLIP features of the 300 Test photos
r = system(torch.load(REPO / "model.pth", map_location="cpu", weights_only=True), T["b_logits"], T["clip"])
right = r["index"] == y; kept = right & ~r["unknown"]
alone = {"vegetable_model": r["vegetable_model"].argmax(1) == y, "clip_head": r["clip_head"].argmax(1) == y}
score = lambda m: {"n": int(m.sum()), "accuracy": int(right[m].sum()), "kept_correct": int(kept[m].sum()),
                   **{k: int(v[m].sum()) for k, v in alone.items()}}
out = {"all": score(np.ones(len(y), bool)), "without_same_session": score(~same),
       "without_same_session_and_watermark": score(~same & ~wm), "same_session_only": score(same),
       "same_session_by_class": dict(collections.Counter(CLASS_NAMES[c] for c in y[same]))}
print(json.dumps(out, indent=1))
