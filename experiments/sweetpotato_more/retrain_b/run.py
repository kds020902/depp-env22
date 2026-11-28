"""Candidate B, development gates only (PREREG.md). T11 is not touched here.

    python experiments/sweetpotato_more/retrain_b/run.py
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
SP = REPO / "experiments/sweetpotato_more"
SPLIT = REPO / "experiments/t10_split"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SPLIT))
from decision import decide  # noqa: E402
from split import to_t10  # noqa: E402

HEAD_DATA = REPO / "experiments/new_vegetables/v7/head_data.npz"
CLS = {"sweetpotato_fresh": 12, "sweetpotato_rotten": 13}


def take(npz, ids, rows):
    f = np.load(npz)
    idx = {str(i): n for n, i in enumerate(f["ids"])}
    n = np.array([idx[i] for i in ids])
    return (f["X"][n].astype(np.float32), f["b"][n].astype(np.float32),
            np.array([CLS[rows[i]["class_name"]] for i in ids]), [rows[i]["source_group"] for i in ids])


def new_train():
    inat = {r["id"]: r for r in csv.DictReader(open(SP / "collected.csv", encoding="utf-8", newline=""))}
    codex = {r["path"]: r for r in csv.DictReader(open(SP / "codex_collected.csv", encoding="utf-8-sig", newline=""))}
    parts = [take(SP / "features.npz", (SPLIT / "train_ids.txt").read_text(encoding="utf-8").split(), inat),
             take(SP / "codex_features.npz", (SPLIT / "codex_train_ids.txt").read_text(encoding="utf-8").split(), codex)]
    extra = [str(i) for i in np.load(SP / "codex_features_extra.npz")["ids"] if not to_t10(codex[str(i)]["source_group"])]
    parts.append(take(SP / "codex_features_extra.npz", extra, codex))
    X = np.concatenate([p[0] for p in parts]); b = np.concatenate([p[1] for p in parts])
    y = np.concatenate([p[2] for p in parts]); g = sum((p[3] for p in parts), [])
    return X, b, y, g


def build_data(path, X, y):
    d = dict(np.load(HEAD_DATA))
    d["train_X"] = np.concatenate([d["train_X"], X]).astype(np.float32)
    d["train_y"] = np.concatenate([d["train_y"], y]).astype(np.int16)
    d["train_tag"] = np.concatenate([d["train_tag"], np.full(len(y), "new_sp", dtype=d["train_tag"].dtype)])
    np.savez(path, **d)


def train(data_path, out_path):
    r = subprocess.run([sys.executable, str(REPO / "src/train_new_veg_head.py"), "--model", str(REPO / "model.pth"),
                        "--data", str(data_path), "--out", str(out_path), "--tau", "0.8"], capture_output=True, text=True, cwd=REPO / "src")
    if r.returncode:
        raise SystemExit(r.stderr)
    return r.stdout.strip().splitlines()[-1]


def answer(ck, b, X):
    a = decide(torch.tensor(b), ck, torch.tensor(X))
    return a.index.numpy(), a.unknown.numpy()


def main():
    WORK.mkdir(exist_ok=True)
    X, b, y, groups = new_train()
    out = {"new_train": {"n": len(y), "fresh": int((y == 12).sum()), "rotten": int((y == 13).sum())}}
    build_data(WORK / "head_data_b.npz", X, y)
    out["train_log"] = train(WORK / "head_data_b.npz", WORK / "candidate_B.pth")

    # G1: the whole development comparison
    dev = WORK / "compare_on_dev_B.json"
    r = subprocess.run([sys.executable, str(REPO / "tools/compare_on_dev.py"), str(dev), f"B={WORK / 'candidate_B.pth'}"],
                       capture_output=True, text=True, cwd=REPO)
    if r.returncode:
        raise SystemExit(r.stderr)
    d = json.loads(dev.read_text(encoding="utf-8"))
    allk = "_all_labelled"
    g1 = {}
    for tag in ("TV-13", "B"):
        s = d["systems"][tag][allk]
        g1[tag] = {"right": s["_all"]["right"] if "_all" in s else None}
    out["G1_raw_tail"] = r.stdout.strip().splitlines()[-25:]
    out["G1_changes"] = d.get("right_and_wrong_against_TV-13")
    out["G1_summary_systems"] = {t: d["systems"][t][allk] for t in ("TV-13", "B")}

    # G2: grouped CV over the new train photos
    fold = np.array([int(hashlib.sha1(("cv:" + g).encode()).hexdigest()[:8], 16) % 5 for g in groups])
    tv13 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    p13, r13 = answer(tv13, b, X)
    pb, rb = np.zeros_like(p13), np.zeros_like(r13)
    for k in range(5):
        m = fold != k
        build_data(WORK / f"head_data_cv{k}.npz", X[m], y[m])
        train(WORK / f"head_data_cv{k}.npz", WORK / f"cv{k}.pth")
        pb[~m], rb[~m] = answer(torch.load(WORK / f"cv{k}.pth", map_location="cpu", weights_only=True), b[~m], X[~m])
    rot = y == 13
    g2 = {}
    for tag, (p, rr) in (("TV-13", (p13, r13)), ("B", (pb, rb))):
        kk = (p // 2 == y // 2) & ~rr
        g2[tag] = {"kind_kept": int(kk.sum()), "rotten_kind_kept": int(kk[rot].sum()), "class_right_kept": int(((p == y) & ~rr).sum())}
    g2["pass"] = g2["B"]["kind_kept"] > g2["TV-13"]["kind_kept"] and g2["B"]["rotten_kind_kept"] > g2["TV-13"]["rotten_kind_kept"]
    out["G2"] = g2
    (HERE / "dev_result.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
