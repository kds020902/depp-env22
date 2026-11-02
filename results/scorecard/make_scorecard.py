"""Scorecard of the adopted model (TV-12): one table over every photo set, with the errors that matter most counted apart.

    python results/scorecard/make_scorecard.py OUT.json

No new scoring of the blind test sets: T3 and T4 use the answers stored when they were scored; Test 300 and the new internet
241 use the model outputs cached at their last scoring (experiments/watermark/final.py), and the script checks that they give
the recorded totals (285 and 188). The answers are only split by class and by kind of error.

Errors, for photos of the five vegetables:
- rotten -> fresh: a rotten vegetable answered as a fresh one (the costly error: someone may eat it);
- fresh -> rotten: a fresh vegetable answered as a rotten one (food thrown away);
- wrong kind: another vegetable (fresh/rotten aside); turned away: "학습된 데이터가 아닙니다".
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/watermark"))
from common import W, clip_features, dev_sets, system  # noqa: E402
from model_utils import CLASS_NAMES  # noqa: E402

FIVE = CLASS_NAMES[:10]
TURNED_AWAY = "거름"


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def card(labels: list[str], answers: list[str]) -> dict:
    pairs = [(y, a) for y, a in zip(labels, answers) if y in FIVE]
    n = len(pairs); right = sum(y == a for y, a in pairs)
    rotten = [(y, a) for y, a in pairs if y.endswith("_rotten")]; fresh = [(y, a) for y, a in pairs if y.endswith("_fresh")]
    kind = lambda c: c.rsplit("_", 1)[0]
    out = {
        "n": n, "right_and_kept": right, "accuracy": round(right / n, 4) if n else None, "ci95": wilson(right, n),
        "turned_away": sum(a == TURNED_AWAY for _, a in pairs),
        "wrong_kind": sum(a != TURNED_AWAY and kind(a) != kind(y) for y, a in pairs),
        "rotten_n": len(rotten), "rotten_to_fresh": sum(a.endswith("_fresh") for _, a in rotten),
        "fresh_n": len(fresh), "fresh_to_rotten": sum(a.endswith("_rotten") for _, a in fresh),
        "per_class": {c: f"{sum(y == c and a == c for y, a in pairs)}/{sum(y == c for y, _ in pairs)}" for c in FIVE if any(y == c for y, _ in pairs)},
    }
    out["rotten_to_fresh_rate"] = round(out["rotten_to_fresh"] / out["rotten_n"], 4) if out["rotten_n"] else None
    out["fresh_to_rotten_rate"] = round(out["fresh_to_rotten"] / out["fresh_n"], 4) if out["fresh_n"] else None
    confusion = {}
    for y, a in pairs:
        confusion.setdefault(y, {}); confusion[y][a] = confusion[y].get(a, 0) + 1
    out["confusion"] = confusion
    return out


def answer_names(r) -> list[str]:
    return [TURNED_AWAY if u else CLASS_NAMES[i] for i, u in zip(r["index"], r["unknown"])]


def main():
    ck = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    res = {"model": ck.get("version", "TV-12")}
    # development photos (seen during development; reported for completeness)
    sets = dev_sets(); F = clip_features(sets); B = dict(np.load(W / "b_WC.npz"))
    for name, label in (("valid", "Valid 300"), ("odev", "first internet 221"), ("o160", "160 field / phone"), ("asian37", "Asian cucumbers 37")):
        res[f"dev:{name}"] = {"label": label, **card([CLASS_NAMES[i] for i in sets[name][1]], answer_names(system(ck, B[name], F[name])))}
    # Test 300 and new internet 241: outputs cached at their last scoring, same answers split by class
    spec = importlib.util.spec_from_file_location("wm_final", REPO / "experiments/watermark/final.py")
    wm = importlib.util.module_from_spec(spec); spec.loader.exec_module(wm)
    ph = wm.photo_sets(); Fe = dict(np.load(W / "final_clip.npz")); Be = dict(np.load(W / "final_b_WC.npz"))
    for name, recorded in (("test_300", 285), ("new_test_241", 188)):
        c = card(ph[name][1], answer_names(system(ck, Be[name], Fe[name])))
        assert c["right_and_kept"] == recorded, (name, c["right_and_kept"], recorded)
        res[f"blind:{name}"] = c
    # T3 and T4: answers stored when scored
    t3 = json.load(open(REPO / "results/t3_test/scores_once.json"))["ZS8"]["answers"]
    res["blind:t3"] = card([r["class_name"] for r in csv.DictReader(open(REPO / "results/t3_test/t3_test.csv", encoding="utf-8"))], t3)
    t4 = json.load(open(REPO / "results/t4_test/scores_reference.json"))["TV-12"]["answers"]
    res["blind:t4"] = card([r["class_name"] for r in csv.DictReader(open(REPO / "results/t4_test/t4_test.csv", encoding="utf-8"))], t4)
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    for k, v in res.items():
        if isinstance(v, dict):
            print(k, v["n"], v["right_and_kept"], v["accuracy"], v["ci95"], "rot->fresh", f"{v['rotten_to_fresh']}/{v['rotten_n']}",
                  "fresh->rot", f"{v['fresh_to_rotten']}/{v['fresh_n']}", "kind", v["wrong_kind"], "away", v["turned_away"])


if __name__ == "__main__":
    main()
