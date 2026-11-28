"""T10, scored ONCE (PREREG.md / PREREG_A2.md): TV-13 against candidate A2 on the 120 sealed photos.

    python experiments/sweetpotato_more/retrain_a/t10.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
from decision import decide  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402

OUT = HERE / "t10_result.json"
if OUT.exists():  # not an assert: python -O would drop it (Codex review)
    raise SystemExit("T10 was already scored once (t10_result.json); it is not scored again")
SP = REPO / "experiments/sweetpotato_more"
CLS = {"sweetpotato_fresh": 12, "sweetpotato_rotten": 13}
rows = {r["id"]: r for r in csv.DictReader(open(SP / "collected.csv", encoding="utf-8", newline=""))}
f = np.load(SP / "features.npz")
idx = {i: n for n, i in enumerate(f["ids"])}
ids = (REPO / "experiments/t10_split/t10_ids.txt").read_text(encoding="utf-8").split()
n = np.array([idx[i] for i in ids])
X, b = torch.tensor(f["X"][n]), torch.tensor(f["b"][n])
y = np.array([CLS[rows[i]["class_name"]] for i in ids])
sprout = np.array(["싹" in rows[i]["note"] for i in ids]) & (y == 13)
rot = y == 13

res = {"n": len(y), "fresh": int((~rot).sum()), "rotten": int(rot.sum()), "rotten_sprout": int(sprout.sum()),
       "rotten_mold_or_rot": int((rot & ~sprout).sum())}
for tag, path in (("TV-13", REPO / "model.pth"), ("A2", HERE / "work/candidate_A2.pth")):
    a = decide(b, torch.load(path, map_location="cpu", weights_only=True), X)
    p, r = a.index.numpy(), a.unknown.numpy()
    kk = (p // 2 == y // 2) & ~r
    right = (p == y) & ~r
    res[tag] = {"kind_kept": int(kk.sum()), "rotten_kind_kept": int(kk[rot].sum()), "fresh_kind_kept": int(kk[~rot].sum()),
                "class_right_kept": int(right.sum()), "rotten_class_right": int(right[rot].sum()),
                "sprout_kind_kept": int(kk[sprout].sum()), "mold_rot_kind_kept": int(kk[rot & ~sprout].sum()),
                "rejected": int(r.sum()),
                "wrong_answers": {f"{CLASS_NAMES[t]}->{'REJ' if j else CLASS_NAMES[q]}": 0 for t, q, j in []}}
    wrong = {}
    for t, q, j in zip(y, p, r):
        if not (q // 2 == t // 2 and not j):
            k = f"{CLASS_NAMES[t]}->{'REJ' if j else CLASS_NAMES[q]}"
            wrong[k] = wrong.get(k, 0) + 1
    res[tag]["wrong_answers"] = dict(sorted(wrong.items(), key=lambda kv: -kv[1]))
t, c = res["TV-13"], res["A2"]
res["adopt_A2"] = bool(c["kind_kept"] >= t["kind_kept"] and c["rotten_kind_kept"] > t["rotten_kind_kept"]
                       and c["fresh_kind_kept"] >= t["fresh_kind_kept"] - 2)
with open(OUT, "x", encoding="utf-8") as fh:  # "x": never overwrite a stored result
    fh.write(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps(res, indent=1, ensure_ascii=False))
