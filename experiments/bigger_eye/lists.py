"""Photo lists for experiments/bigger_eye: every photo whose CLIP features TV-12 uses, by path, so that another image model can be
put in the same place. The rows of the new-vegetable head (experiments/new_vegetables/v7/head_data.npz keeps features only) are
found by matching the stored ViT-B-16 features against features with known paths (cosine >= 0.999).

    python experiments/bigger_eye/lists.py      # writes $DEPP_WORK/wm/eye_lists.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "watermark"))
from common import DEV_LISTS, REPO, S, W, photo_sets  # noqa: E402


def main():
    sets = json.load(open(S / "clip/sets.json"))
    pool = json.load(open(S / "v7/pool.json"))
    dev = {n: [p for p, _ in sets[k]] for n, k in (("valid", "valid"), ("odev", "odev"), ("o160", "o160"), ("asian37", "asian37"),
                                                    ("fruit374", "fruit374"), ("coco400", "coco400"), ("lvis", "lvis_veg"))}
    dev["newval"] = [r["path"] for r in pool if r["split"] == "val"]

    # rotten fruit the head should turn away, in the order of common.rotten_fruit_features()
    rows = list(csv.DictReader(open(S / "arb/pool/pool.csv", encoding="utf-8"))); keep = set(np.load(S / "arb/pool/keep_idx.npy").tolist())
    rotten = {"kaggle_fruit": [r["path"] for i, r in enumerate(rows) if i in keep and r["group"] == "kaggle_fruit"],
              "inat_rotten_fruit": [str(S / f"arb/inat/img/{i.removeprefix('inat-')}.jpg") for i in json.load(open(S / "arb/pool/inat_train_keep.json"))],
              "openverse_rotten_fruit": [str(S / f"arb/img/{i}.jpg") for i in np.load(S / "arb/ov_train_fruit.npz")["ids"]]}

    # the new-vegetable head's 6,271 training rows: match the stored features to features with known paths
    bank_paths, bank = [], []
    h = np.load(W / "head_features.npz")
    plain = [i for i, k in enumerate(h["keys"]) if not str(k).startswith("watermark")]
    bank_paths += [str(h["keys"][i]) for i in plain]; bank.append(h["features"][plain])
    bank_paths += [r["path"] for r in pool]; bank.append(np.load(S / "v7/pool_feats.npz")["f"])
    for k in ("fruit", "everyday", "otherveg"):
        z = np.load(S / f"neg/{k}_feats.npz"); bank_paths += [str(p) for p in z["paths"]]; bank.append(z["f"])
    bank = torch.nn.functional.normalize(torch.tensor(np.concatenate(bank)).float(), dim=1)
    data = np.load(REPO / "experiments/new_vegetables/v7/head_data.npz")
    x = torch.nn.functional.normalize(torch.tensor(data["train_X"]).float(), dim=1)
    sims, idx = [], []
    for i in range(0, len(x), 512):
        s, j = (x[i:i + 512] @ bank.T).max(1); sims.append(s); idx.append(j)
    sims, idx = torch.cat(sims), torch.cat(idx)
    newveg = [[bank_paths[int(j)], int(y)] for j, y in zip(idx, data["train_y"])]
    print("new-vegetable rows matched:", len(newveg), "lowest cosine", round(float(sims.min()), 5), "below 0.999:", int((sims < 0.999).sum()))
    assert float(sims.min()) >= 0.999

    t5 = [r["path"] for r in csv.DictReader(open(DEV_LISTS / "dev.csv", encoding="utf-8"))]
    ph = photo_sets()
    out = {"dev": dev, "rotten_fruit": rotten, "newveg_train": newveg, "t5_dev": t5,
           "test_300": ph["test_300"][0], "new_test_241": ph["new_test_241"][0]}
    missing = [p for v in [*dev.values(), *rotten.values(), [p for p, _ in newveg], t5] for p in v if not Path(p).exists()]
    print({k: (len(v) if isinstance(v, list) else {kk: len(vv) for kk, vv in v.items()}) for k, v in out.items()}, "missing files:", len(missing))
    assert not missing, missing[:5]
    json.dump(out, open(W / "eye_lists.json", "w"))


if __name__ == "__main__":
    main()
