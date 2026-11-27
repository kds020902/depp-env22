"""Split Codex's recovered sweet potato photos (codex_features.npz) with the SAME rule as split.py, before any use.

    python experiments/t10_split/split_codex.py

- T10 stays as it is (120 photos, already scored once). Photos of Codex's list whose source_group hashes to the test side
  form a separate new test, T11; they are never added to T10.
- A training photo with CLIP cosine >= 0.95 to any T10 or T11 photo is dropped (near duplicate of a test photo).
Writes t11_ids.txt / codex_train_ids.txt and codex_counts.json.
"""

from __future__ import annotations

import collections
import csv
import json
from pathlib import Path

import numpy as np

from split import to_t10

HERE = Path(__file__).resolve().parent
SP = HERE.parents[0] / "sweetpotato_more"


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def main():
    rows = {r["path"]: r for r in csv.DictReader(open(SP / "codex_collected.csv", encoding="utf-8-sig", newline=""))}
    f = np.load(SP / "codex_features.npz")
    ids = [str(i) for i in f["ids"]]
    X = unit(f["X"].astype(np.float32))
    assert all(rows[i]["source_group"].strip() for i in ids)
    test = np.array([to_t10(rows[i]["source_group"]) for i in ids])
    g = np.load(SP / "features.npz")
    t10 = set((HERE / "t10_ids.txt").read_text(encoding="utf-8").split())
    T10 = unit(g["X"][[n for n, i in enumerate(g["ids"]) if i in t10]].astype(np.float32))
    near = (X @ np.concatenate([T10, X[test]]).T).max(1) >= 0.95
    drop = ~test & near
    t11 = sorted(i for i, t in zip(ids, test) if t)
    train = sorted(i for i, t, d in zip(ids, test, drop) if not t and not d)
    count = collections.Counter()
    for i, t, d in zip(ids, test, drop):
        side = "T11" if t else ("dropped_near_test" if d else "train")
        sub = "sprout" if "싹" in rows[i]["severity"] + rows[i]["note"] else "other"
        count[f"{side}/{rows[i]['class_name']}" + (f"/{sub}" if rows[i]["class_name"].endswith("rotten") else "")] += 1
    (HERE / "t11_ids.txt").write_text("\n".join(t11) + "\n", encoding="utf-8")
    (HERE / "codex_train_ids.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    out = dict(sorted(count.items()))
    (HERE / "codex_counts.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
