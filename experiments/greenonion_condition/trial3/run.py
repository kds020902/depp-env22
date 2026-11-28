"""Trial 3, development only (PREREG.md; trial 2 photos and settings, rotten rows weighted to balance fresh, integrity checks first): green onion condition learned by the second head. The final test is not read.

    python experiments/greenonion_condition/trial3/run.py
Uses trial3/photos.csv and trial3/features.npz, made by
    python experiments/sweetpotato_more/features_from_folder.py experiments/greenonion_condition/trial3/photos.csv D:/AI experiments/greenonion_condition/trial3/features.npz
Writes result.json; work files in work/ (not committed).
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
sys.path.insert(0, str(HERE))
from integrity import problems  # noqa: E402

HEAD_DATA = REPO / "experiments/new_vegetables/v7/head_data.npz"
GO_FRESH, GO_ROTTEN = 14, 15


def build_data(path, X, y):
    d = dict(np.load(HEAD_DATA))
    d["train_X"] = np.concatenate([d["train_X"], X]).astype(np.float32)
    d["train_y"] = np.concatenate([d["train_y"], y]).astype(np.int16)
    d["train_tag"] = np.concatenate([d["train_tag"], np.full(len(y), "greenonion_user", dtype=d["train_tag"].dtype)])
    np.savez(path, **d)


def train(data_path, out_path):
    r = subprocess.run([sys.executable, str(HERE / "train15w.py"), "--model", str(REPO / "model.pth"),
                        "--data", str(data_path), "--out", str(out_path), "--tau", "0.8"], capture_output=True, text=True, cwd=REPO / "src")
    if r.returncode:
        raise SystemExit(r.stderr)
    return r.stdout.strip().splitlines()[-1]


def answer(ck, b, X):
    a = decide(torch.tensor(b), ck, torch.tensor(X))
    return a.index.numpy(), a.unknown.numpy()


def tally(p, rej, cond):
    green = (p // 2 == 7) & ~rej
    out = {}
    for c, want in (("spoiled", GO_ROTTEN), ("fresh", GO_FRESH)):
        m = cond == c
        out[c] = {"n": int(m.sum()), "recognised": int(green[m].sum()), "turned_away": int(rej[m].sum()),
                  "condition_right": int((green & (p == want))[m].sum())}
    return out


def main():
    bad = problems()  # stop before training if any photo, list or feature cache does not match
    if bad:
        raise SystemExit("integrity FAILED:\n  " + "\n  ".join(bad))
    WORK.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(HERE / "photos.csv", encoding="utf-8")))
    f = np.load(HERE / "features.npz")
    idx = {str(i): n for n, i in enumerate(f["ids"])}
    missing = [r["path"] for r in rows if r["path"] not in idx]
    if missing:
        raise SystemExit(f"features missing for {len(missing)} photos, e.g. {missing[:3]}")
    n = np.array([idx[r["path"]] for r in rows])
    X, b = f["X"][n].astype(np.float32), f["b"][n].astype(np.float32)
    cond = np.array([r["condition"] for r in rows])
    y = np.where(cond == "spoiled", GO_ROTTEN, GO_FRESH)
    groups = [r["group"] for r in rows]
    fold = np.array([int(hashlib.sha1(("go:" + g).encode()).hexdigest()[:8], 16) % 5 for g in groups])

    tv13 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    out = {"n_photos": len(rows), "n_groups": len(set(groups)), "tv13": tally(*answer(tv13, b, X), cond)}

    pa, ra = np.zeros(len(rows), int), np.zeros(len(rows), bool)
    for k in range(5):
        m = fold != k
        build_data(WORK / f"cv{k}.npz", X[m], y[m])
        train(WORK / f"cv{k}.npz", WORK / f"cv{k}.pth")
        pa[~m], ra[~m] = answer(torch.load(WORK / f"cv{k}.pth", map_location="cpu", weights_only=True), b[~m], X[~m])
    out["candidate_cv"] = tally(pa, ra, cond)

    build_data(WORK / "all.npz", X, y)
    out["train_log"] = train(WORK / "all.npz", WORK / "candidate.pth")
    dev = WORK / "compare_on_dev.json"
    r = subprocess.run([sys.executable, str(REPO / "tools/compare_on_dev.py"), str(dev), f"C={WORK / 'candidate.pth'}"], capture_output=True, text=True, cwd=REPO)
    if r.returncode:
        raise SystemExit(r.stderr)
    d = json.loads(dev.read_text(encoding="utf-8"))["systems"]
    s13, sc = d["TV-13"]["_all_labelled"], d["C"]["_all_labelled"]
    g1 = {"right": (s13["right"], sc["right"]), "rotten_called_fresh": (s13["rotten_called_fresh"], sc["rotten_called_fresh"]),
          "potato_rotten": (s13["classes"]["potato_rotten"]["right"], sc["classes"]["potato_rotten"]["right"]),
          "potato_fresh": (s13["classes"]["potato_fresh"]["right"], sc["classes"]["potato_fresh"]["right"])}
    out["G1"] = {**{k: {"TV-13": v[0], "C": v[1]} for k, v in g1.items()},
                 "pass": sc["right"] >= 869 and g1["potato_rotten"][1] >= 126 and g1["potato_fresh"][1] >= 107 and sc["rotten_called_fresh"] <= 17}
    t, c = out["tv13"]["spoiled"], out["candidate_cv"]
    out["G2"] = {"TV-13": t["recognised"], "C": c["spoiled"]["recognised"], "pass": c["spoiled"]["recognised"] > t["recognised"]}
    rate = lambda s: s["condition_right"] / s["recognised"] if s["recognised"] else 0.0  # noqa: E731
    out["G3"] = {"spoiled": round(rate(c["spoiled"]), 3), "fresh": round(rate(c["fresh"]), 3),
                 "pass": rate(c["spoiled"]) >= 0.6 and rate(c["fresh"]) >= 0.6}
    (HERE / "result.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
