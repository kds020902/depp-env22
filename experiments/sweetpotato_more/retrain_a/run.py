"""Candidate A, development gates only (PREREG.md). T10 is not touched here.

    python experiments/sweetpotato_more/retrain_a/run.py

Builds head_data + the 168 T10-split train photos, trains candidate.pth with src/train_new_veg_head.py (unchanged),
then D1 (old vegetables), D2 (new-vegetable Valid 89) and D3 (grouped 5-fold CV over the 168) against TV-13.
Writes dev_result.json. Work files go to retrain_a/work/ (not committed).
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WORK = HERE / "work"
sys.path.insert(0, str(REPO / "src"))
from decision import decide  # noqa: E402

HEAD_DATA = REPO / "experiments/new_vegetables/v7/head_data.npz"
SP = REPO / "experiments/sweetpotato_more"
CLS = {"sweetpotato_fresh": 12, "sweetpotato_rotten": 13}


def load_new():
    rows = {r["id"]: r for r in csv.DictReader(open(SP / "collected.csv", encoding="utf-8", newline=""))}
    f = np.load(SP / "features.npz")
    idx = {i: n for n, i in enumerate(f["ids"])}
    return rows, f, idx


def subset(ids, rows, f, idx):
    n = np.array([idx[i] for i in ids])
    y = np.array([CLS[rows[i]["class_name"]] for i in ids])
    g = [rows[i]["source_group"] for i in ids]
    return f["X"][n].astype(np.float32), f["b"][n].astype(np.float32), y, g


def build_data(path, X, y):
    d = dict(np.load(HEAD_DATA))
    d["train_X"] = np.concatenate([d["train_X"], X]).astype(np.float32)
    d["train_y"] = np.concatenate([d["train_y"], y]).astype(np.int16)
    d["train_tag"] = np.concatenate([d["train_tag"], np.full(len(y), "inat_train", dtype=d["train_tag"].dtype)])
    np.savez(path, **d)


def train(data_path, out_path):
    r = subprocess.run([sys.executable, str(REPO / "src/train_new_veg_head.py"), "--model", str(REPO / "model.pth"),
                        "--data", str(data_path), "--out", str(out_path)], capture_output=True, text=True, cwd=REPO / "src")
    if r.returncode:
        raise SystemExit(r.stderr)
    return r.stdout.strip().splitlines()[-1]


def answer(ck, b, X):
    a = decide(torch.tensor(b), ck, torch.tensor(X))
    return a.index.numpy(), a.unknown.numpy()


def kind_kept(p, rej, y):
    return (p // 2 == y // 2) & ~rej


def main():
    WORK.mkdir(exist_ok=True)
    rows, f, idx = load_new()
    train_ids = (REPO / "experiments/t10_split/train_ids.txt").read_text(encoding="utf-8").split()
    X, b, y, groups = subset(train_ids, rows, f, idx)
    tv13 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)

    build_data(WORK / "head_data_a.npz", X, y)
    log = train(WORK / "head_data_a.npz", WORK / "candidate.pth")
    cand = torch.load(WORK / "candidate.pth", map_location="cpu", weights_only=True)
    out = {"train_log": log, "tv13_tau": tv13["new_vegetable_head"]["switch_probability"],
           "cand_tau": cand["new_vegetable_head"]["switch_probability"]}

    d = np.load(HEAD_DATA)
    # D1
    d1 = {}
    for name in ("valid", "odev", "o160", "asian37"):
        yy = d[f"dev_{name}_y"].astype(int)
        d1[name] = {}
        for tag, ck in (("TV-13", tv13), ("A", cand)):
            p, rej = answer(ck, d[f"dev_{name}_b"], d[f"dev_{name}_X"])
            d1[name][tag] = int(((p == yy) & ~rej).sum())
    for name in ("fruit374", "coco400", "lvis"):
        d1[name] = {}
        for tag, ck in (("TV-13", tv13), ("A", cand)):
            _, rej = answer(ck, d[f"dev_{name}_b"], d[f"dev_{name}_X"])
            d1[name][tag] = int(rej.sum())
    kept = lambda t: sum(d1[n][t] for n in ("valid", "odev", "o160", "asian37"))
    rejd = lambda t: sum(d1[n][t] for n in ("fruit374", "coco400", "lvis"))
    out["D1"] = {**d1, "pass": kept("A") >= kept("TV-13") - 2 and rejd("A") >= rejd("TV-13") - 3}
    # D2
    yy = d["dev_newval_y"].astype(int)
    d2 = {}
    for tag, ck in (("TV-13", tv13), ("A", cand)):
        p, rej = answer(ck, d["dev_newval_b"], d["dev_newval_X"])
        d2[tag] = int(kind_kept(p, rej, yy).sum())
    out["D2"] = {**d2, "of": len(yy), "pass": d2["A"] >= d2["TV-13"]}
    # D3: grouped CV over the 168
    fold = np.array([int(hashlib.sha1(("cv:" + g).encode()).hexdigest()[:8], 16) % 5 for g in groups])
    p13, r13 = answer(tv13, b, X)
    pa, ra = np.zeros_like(p13), np.zeros_like(r13)
    for k in range(5):
        m = fold != k
        build_data(WORK / f"head_data_cv{k}.npz", X[m], y[m])
        train(WORK / f"head_data_cv{k}.npz", WORK / f"cv{k}.pth")
        ckk = torch.load(WORK / f"cv{k}.pth", map_location="cpu", weights_only=True)
        pa[~m], ra[~m] = answer(ckk, b[~m], X[~m])
    rot = y == 13
    d3 = {}
    for tag, (p, r) in (("TV-13", (p13, r13)), ("A", (pa, ra))):
        kk = kind_kept(p, r, y)
        d3[tag] = {"kind_kept": int(kk.sum()), "rotten_kind_kept": int(kk[rot].sum()), "fresh_kind_kept": int(kk[~rot].sum()),
                   "class_right_kept": int(((p == y) & ~r).sum()), "rejected": int(r.sum())}
    out["D3"] = {**d3, "of": len(y), "rotten_of": int(rot.sum()),
                 "pass": d3["A"]["kind_kept"] > d3["TV-13"]["kind_kept"] and d3["A"]["rotten_kind_kept"] > d3["TV-13"]["rotten_kind_kept"]}
    out["open_T10"] = bool(out["D1"]["pass"] and out["D2"]["pass"] and out["D3"]["pass"])
    (HERE / "dev_result.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
