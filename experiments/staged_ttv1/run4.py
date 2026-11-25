"""Logit adjustment (LA) and a Tip-Adapter style cache (TIP) on TV-13's second CLIP head (PREREG.md, addendum 3).

    python experiments/staged_ttv1/run4.py experiments/staged_ttv1/results4.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools"))
import decision  # noqa: E402
from compare_on_dev import AWAY, UNLABELLED, answers_of, development_sets, right, says_fresh  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402

ORIGINAL = decision.new_vegetable_logits
ALPHA, BETA, SHOTS = 1.0, 5.5, 16


def scratch_sets():
    d = np.load(HERE / "scratch_features.npz")
    out = {}
    for g in sorted(set(d["group"].tolist())):
        m = d["group"] == g
        out[f"scratch: {g}"] = (torch.tensor(d["b"][m]), torch.tensor(d["X"][m]), [CLASS_NAMES[int(i)] for i in d["y"][m]])
    return out


def with_logit_adjustment(tau, counts):
    def logits(features, head):
        z = ORIGINAL(features, head).clone()
        for c, n in counts.items():
            z[:, c] -= tau * np.log(n / sum(counts.values()))
        return z
    return logits


def with_cache(train_X, train_y):
    keys = {c: torch.tensor(train_X[train_y == c]) for c in sorted(set(train_y.tolist()))}

    def logits(features, head):
        z = ORIGINAL(features, head).clone()
        for c, k in keys.items():
            affinity = torch.exp(-BETA * (1 - features.float() @ k.T))
            z[:, c] += ALPHA * affinity.topk(min(SHOTS, k.shape[0]), 1).values.sum(1)
        return z
    return logits


def run(checkpoint, sets):
    return {name: answers_of(checkpoint, b, f)[0] for name, (b, f, _) in sets.items()}


def summarise(answers, sets, keep=None):
    """Counts over the labelled photos (optionally only the photos `keep[name]` marks), and the unlabelled sets."""
    out = {"right": 0, "n": 0, "rotten_called_fresh": 0, "classes": {}, "should_turn_away": 0, "should_keep": 0}
    for name, (_, _, labels) in sets.items():
        a = answers[name]
        if labels is None:
            reject = dict(UNLABELLED.values())[name]
            out["should_turn_away" if reject else "should_keep"] += a.count(AWAY) if reject else len(a) - a.count(AWAY)
            continue
        for i, (label, x) in enumerate(zip(labels, a)):
            if keep is not None and not keep[name][i]:
                continue
            c = out["classes"].setdefault(label, [0, 0])
            c[0] += right(label, x); c[1] += 1
            out["right"] += right(label, x); out["n"] += 1
            out["rotten_called_fresh"] += label.endswith("_rotten") and says_fresh(x)
    return out


def flips(base, other, sets):
    lost = gained = 0
    for name, (_, _, labels) in sets.items():
        for label, x, y in zip(labels or [], base[name], other[name]):
            lost += right(label, x) and not right(label, y)
            gained += not right(label, x) and right(label, y)
    return {"right_to_wrong": int(lost), "wrong_to_right": int(gained)}


def main():
    out = Path(sys.argv[1])
    checkpoint = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    data = np.load(REPO / "experiments/new_vegetables/v7/head_data.npz", allow_pickle=True)
    train_X, train_y = data["train_X"], data["train_y"].astype(int)
    counts = {int(c): int((train_y == c).sum()) for c in set(train_y.tolist())}
    dev, scratch = development_sets(), scratch_sets()
    train = torch.tensor(train_X)
    far = {name: ((f.float() @ train.T).max(1).values < 0.95).tolist() for name, (_, f, _) in dev.items()}
    systems = {"TV-13": ORIGINAL, "LA tau=1": with_logit_adjustment(1.0, counts), "LA tau=0.5": with_logit_adjustment(0.5, counts),
               "TIP": with_cache(train_X, train_y)}
    record, answers = {"class_counts": {CLASS_NAMES[c] if c < 16 else "other": n for c, n in sorted(counts.items())},
                       "near_training_rows_left_out": {n: len(v) - sum(v) for n, v in far.items()}, "systems": {}}, {}
    for name, function in systems.items():
        decision.new_vegetable_logits = function
        answers[name] = {"dev": run(checkpoint, dev), "scratch": run(checkpoint, scratch)}
        d, s = summarise(answers[name]["dev"], dev), summarise(answers[name]["scratch"], scratch)
        rotten_sweet = [d["classes"]["sweetpotato_rotten"][0] + s["classes"]["sweetpotato_rotten"][0],
                        d["classes"]["sweetpotato_rotten"][1] + s["classes"]["sweetpotato_rotten"][1]]
        record["systems"][name] = {"dev": d, "dev_without_near_training_rows": summarise(answers[name]["dev"], dev, far), "scratch": s,
                                   "rotten_sweet_potato_dev_and_scratch": rotten_sweet,
                                   "per_scratch_group": {g: summarise({g: answers[name]["scratch"][g]}, {g: scratch[g]})["classes"] for g in scratch},
                                   "dev_against_TV-13": flips(answers["TV-13"]["dev"], answers[name]["dev"], dev),
                                   "scratch_against_TV-13": flips(answers["TV-13"]["scratch"], answers[name]["scratch"], scratch)}
    decision.new_vegetable_logits = ORIGINAL
    base = record["systems"]["TV-13"]
    for name, r in record["systems"].items():
        r["criteria"] = {"1 rotten sweet potato +8": r["rotten_sweet_potato_dev_and_scratch"][0] - base["rotten_sweet_potato_dev_and_scratch"][0] >= 8,
                         "2 dev right->wrong <= wrong->right": r["dev_against_TV-13"]["right_to_wrong"] <= r["dev_against_TV-13"]["wrong_to_right"],
                         "3 turned away not down by more than 3": base["dev"]["should_turn_away"] - r["dev"]["should_turn_away"] <= 3,
                         "4 rotten called fresh not up": r["dev"]["rotten_called_fresh"] <= base["dev"]["rotten_called_fresh"]}
    json.dump(record, open(out, "w"), indent=1, ensure_ascii=False)
    for name, r in record["systems"].items():
        d, s = r["dev"], r["scratch"]
        print(f"{name:11s} dev {d['right']}/{d['n']} (far {r['dev_without_near_training_rows']['right']}/{r['dev_without_near_training_rows']['n']}) r->f {d['rotten_called_fresh']} away {d['should_turn_away']} kept {d['should_keep']} "
              f"| scratch {s['right']}/{s['n']} r->f {s['rotten_called_fresh']} | rotten sp {r['rotten_sweet_potato_dev_and_scratch']} "
              f"| sp_fresh dev {d['classes']['sweetpotato_fresh']} scr {s['classes']['sweetpotato_fresh']} potato_fresh dev {d['classes']['potato_fresh']} scr {s['classes']['potato_fresh']} potato_rotten scr {s['classes']['potato_rotten']} "
              f"| flips dev {r['dev_against_TV-13']} scr {r['scratch_against_TV-13']} | {list(r['criteria'].values())}")


if __name__ == "__main__":
    main()
