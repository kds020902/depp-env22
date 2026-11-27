"""Can the second head's CLIP features separate the hard pairs? See PREREG.md.

    python experiments/feature_probe/run.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[0] / "new_vegetables" / "v7" / "head_data.npz"
PAIRS = {"P1": ([2, 3], [12, 13]), "P2": ([3], [13]), "P3": ([12], [13]), "P4": ([2], [3])}
SETS = ["train", "dev_valid", "dev_odev", "dev_o160", "dev_newval"]
DUP = 0.95
FOLDS = 5
SEEDS = [0, 1, 2]


def load():
    d = np.load(DATA)
    X = np.concatenate([d[f"{s}_X"] for s in SETS]).astype(np.float32)
    y = np.concatenate([d[f"{s}_y"] for s in SETS]).astype(int)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    T = d["text16"].astype(np.float32)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    return X, y, T


def dup_groups(X):
    n = len(X)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    S = X @ X.T
    for i, j in zip(*np.where(np.triu(S, 1) >= DUP)):
        parent[find(i)] = find(j)
    return np.array([find(i) for i in range(n)])


def group_folds(groups, y, seed):
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), int)
    for cls in (0, 1):  # deal groups of each class round-robin so folds stay balanced
        gs = np.unique(groups[y == cls])
        rng.shuffle(gs)
        for k, g in enumerate(gs):
            fold[(groups == g) & (y == cls)] = k % FOLDS
    # a group mixing classes must sit in one fold
    for g in np.unique(groups):
        m = groups == g
        fold[m] = fold[m][0]
    return fold


def logreg(Xtr, ytr, Xte):
    Xt = torch.tensor(Xtr)
    yt = torch.tensor(ytr, dtype=torch.float32)
    w_pos = len(ytr) / (2 * max(ytr.sum(), 1))
    w_neg = len(ytr) / (2 * max((1 - ytr).sum(), 1))
    sw = torch.where(yt == 1, w_pos, w_neg)
    w = torch.zeros(Xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")
    scale = 20.0  # unit-norm features need a temperature to reach confident logits

    def closure():
        opt.zero_grad()
        z = scale * (Xt @ w) + b
        loss = (sw * torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none")).mean()
        loss = loss + 1e-3 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        return ((scale * (torch.tensor(Xte) @ w) + b) > 0).numpy().astype(int)


def knn(Xtr, ytr, Xte, k=10):
    S = Xte @ Xtr.T
    idx = np.argsort(-S, axis=1)[:, :k]
    votes = ytr[idx]
    # class-balanced vote
    p1 = votes.mean(1) / max(ytr.mean(), 1e-9)
    p0 = (1 - votes).mean(1) / max(1 - ytr.mean(), 1e-9)
    return (p1 > p0).astype(int)


def metrics(y, p):
    r1 = float((p[y == 1] == 1).mean())
    r0 = float((p[y == 0] == 0).mean())
    return {"bal_acc": round((r0 + r1) / 2, 4), "recall_B": round(r1, 4), "recall_A": round(r0, 4)}


def main():
    X, y, T = load()
    out = {}
    for name, (A, B) in PAIRS.items():
        m = np.isin(y, A + B)
        Xs, ys = X[m], np.isin(y[m], B).astype(int)
        groups = dup_groups(Xs)
        res = {"n_A": int((ys == 0).sum()), "n_B": int((ys == 1).sum()), "groups": int(len(np.unique(groups)))}
        # zero-shot: best prompt among A classes vs B classes
        S = Xs @ T.T
        res["zeroshot"] = metrics(ys, (S[:, B].max(1) > S[:, A].max(1)).astype(int))
        for clf_name, clf in (("linear", logreg), ("knn10", knn)):
            runs = []
            for seed in SEEDS:
                fold = group_folds(groups, ys, seed)
                p = np.zeros_like(ys)
                for f in range(FOLDS):
                    te = fold == f
                    p[te] = clf(Xs[~te], ys[~te], Xs[te])
                runs.append(metrics(ys, p))
            res[clf_name] = {k: round(float(np.mean([r[k] for r in runs])), 4) for k in runs[0]}
            res[clf_name]["bal_acc_runs"] = [r["bal_acc"] for r in runs]
        out[name] = res
        print(name, json.dumps(res, ensure_ascii=False))
    (HERE / "result.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
