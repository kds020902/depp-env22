"""Part 4 of PREREG.md: answer "판단 보류" (hold the answer back) when the probability of the chosen answer is below q.
Risk-coverage numbers per development set for q = 0.5 ... 0.9, and q chosen by the pre-registered rule: the smallest q at which
the answered photos of the outside development sets (internet 221 + 160 + Asian cucumbers 37) are 95% or more right; if none,
the present retake threshold 0.9. Only photos the system does not turn away are counted (turned-away photos are answered
"학습된 데이터가 아닙니다" as before).

    python experiments/improve_without_photos/abstain.py MODEL OUT.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "watermark"))
from common import W, clip_features, dev_sets  # noqa: E402
from decision import decide  # noqa: E402

QS = (0.5, 0.6, 0.7, 0.8, 0.9)
SETS = ("valid", "odev", "o160", "asian37")
OUTSIDE = ("odev", "o160", "asian37")


def table(right: np.ndarray, conf: np.ndarray, same_pair: np.ndarray) -> dict:
    rows = {"answered_all": {"answered": int(len(right)), "right": int(right.sum()), "accuracy": round(float(right.mean()), 4) if len(right) else None}}
    for q in QS:
        a = conf >= q; held = ~a
        rows[f"q{q}"] = {"answered": int(a.sum()), "coverage": round(float(a.mean()), 4) if len(a) else None,
                         "right_of_answered": int(right[a].sum()), "accuracy_answered": round(float(right[a].mean()), 4) if a.any() else None,
                         "held_back": int(held.sum()), "held_back_wrong": int((~right[held]).sum()),
                         "held_back_fresh_or_rotten_doubt": int((held & same_pair).sum())}
    return rows


def main():
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    sets = dev_sets(); F = clip_features(sets); B = dict(np.load(W / "b_WC.npz"))
    res, pooled = {"model": Path(sys.argv[1]).name}, {"right": [], "conf": [], "pair": []}
    for name in SETS:
        a = decide(torch.tensor(B[name]), ck, torch.tensor(F[name])); y = sets[name][1]
        index, conf, unknown = a.index.numpy(), a.probability.numpy(), a.unknown.numpy()
        top2 = np.argsort(-a.five.numpy(), 1)[:, :2]
        same_pair = (top2[:, 0] // 2) == (top2[:, 1] // 2)          # doubt is between fresh and rotten of one vegetable
        keep = ~unknown
        right = (index == y)[keep]
        res[name] = table(right, conf[keep], same_pair[keep])
        if name in OUTSIDE:
            pooled["right"].append(right); pooled["conf"].append(conf[keep]); pooled["pair"].append(same_pair[keep])
    res["outside_pooled"] = table(*(np.concatenate(pooled[k]) for k in ("right", "conf", "pair")))
    good = [q for q in QS if (res["outside_pooled"][f"q{q}"]["accuracy_answered"] or 0) >= 0.95]
    res["chosen_q"] = min(good) if good else 0.9
    res["rule"] = "smallest q with >= 95% right among answered outside development photos; else 0.9 (present retake threshold)"
    json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"chosen_q": res["chosen_q"], "outside": res["outside_pooled"]}, indent=1))


if __name__ == "__main__":
    main()
