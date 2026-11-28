"""Green onion condition, step 0: two scores that need no rotten training photo, thresholds set on FRESH photos only.

    python experiments/greenonion_condition/prep.py

- zs   : CLIP zero-shot margin = mean cosine to "spoiled" prompts - mean cosine to "fresh" prompts (same ViT-B-16 as the heads)
- knn  : 1 - mean cosine to the 5 nearest fresh green onion training photos (leave-one-out on the training photos)
Threshold per score: 95% of the fresh green onions (Train 97 + new-vegetable Valid 20) stay "fresh" (like the "not eight" rule).
No rotten green onion photo exists yet, so nothing here says how well rotten ones are caught; that needs a labelled test set.
Writes thresholds.json and text_prompts.npz.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import open_clip
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
GO = 14  # greenonion_fresh
FRESH = ["a photo of fresh green onions.", "a photo of a bunch of fresh scallions.", "a close-up photo of fresh green onions.",
         "a photo of fresh welsh onions."]
SPOILED = ["a photo of wilted yellow green onions.", "a photo of rotten slimy green onions.", "a photo of moldy scallions.",
           "a photo of spoiled spring onions.", "a photo of dried out withered green onions."]


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def main():
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="datacomp_xl_s13b_b90k")
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-16")
    with torch.inference_mode():
        Tf = unit(model.encode_text(tok(FRESH)).numpy())
        Ts = unit(model.encode_text(tok(SPOILED)).numpy())
    np.savez(HERE / "text_prompts.npz", fresh=Tf, spoiled=Ts, fresh_text=np.array(FRESH), spoiled_text=np.array(SPOILED))

    d = np.load(REPO / "experiments/new_vegetables/v7/head_data.npz")
    tr = unit(d["train_X"][d["train_y"] == GO].astype(np.float32))
    va = unit(d["dev_newval_X"][d["dev_newval_y"] == GO].astype(np.float32))
    zs = lambda X: (X @ Ts.T).mean(1) - (X @ Tf.T).mean(1)
    S = tr @ tr.T
    np.fill_diagonal(S, -1)
    knn_tr = 1 - np.sort(S, 1)[:, -5:].mean(1)
    knn_va = 1 - np.sort(va @ tr.T, 1)[:, -5:].mean(1)
    zs_all = np.concatenate([zs(tr), zs(va)])
    knn_all = np.concatenate([knn_tr, knn_va])
    out = {"n_fresh": int(len(zs_all)),
           "zs": {"threshold_95": float(np.quantile(zs_all, 0.95)), "min": float(zs_all.min()), "median": float(np.median(zs_all)),
                  "max": float(zs_all.max())},
           "knn": {"threshold_95": float(np.quantile(knn_all, 0.95)), "min": float(knn_all.min()), "median": float(np.median(knn_all)),
                   "max": float(knn_all.max())},
           "note": "thresholds from fresh photos only; detection of rotten green onions not measured yet (no labelled rotten photos)"}
    (HERE / "thresholds.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
