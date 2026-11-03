"""experiments/no_greenonion (PREREG.md): TV-12 against G1 on the development photos.

    python experiments/no_greenonion/evaluate.py OUT.json $DEPP_WORK/wm/G1.pth

T3, T4, T5 (development now) use the CLIP features and vegetable-model outputs cached by experiments/root_and_tomato/evaluate.py
(computed from the same photos); the keep-rule sets use the features stored in head_data.npz. No blind test set is read.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("rt_eval", REPO / "experiments/root_and_tomato/evaluate.py")
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
AWAY, CLASS_NAMES = rt.AWAY, rt.CLASS_NAMES
DEV = ("t3", "t4", "t5")


def score(ck: dict, sets, B, F) -> dict:
    ans = {n: rt.answers(ck, B[n], F[n]) for n in sets}
    pairs = lambda names: [(y, a) for n in names for y, a in zip(sets[n][1], ans[n])]
    green = [(y, a) for y, a in pairs(DEV + ("newval",)) if y and y.startswith("greenonion")]
    kind = lambda ps: [sum(a != AWAY and a.rsplit("_", 1)[0] == y.rsplit("_", 1)[0] for y, a in ps), len(ps)]
    on_sp = lambda names: [(y, a) for y, a in pairs(names) if y and y.startswith(("onion", "sweetpotato"))]
    kept = lambda n: [sum(a == y for y, a in zip(sets[n][1], ans[n])), len(ans[n])]
    old = sum(a != AWAY and CLASS_NAMES.index(a) >= 10 for n in ("valid", "odev", "asian37", "lvis") for a in ans[n])
    rf = [(y, a) for y, a in pairs(DEV) if y == "rotten_fruit"]
    return {"green_onion_turned_away": [sum(a == AWAY for _, a in green), len(green)],
            "green_onion_answered_as": dict(__import__("collections").Counter(a for _, a in green if a != AWAY)),
            "onion_sweetpotato_kind_dev48": kind(on_sp(DEV)), "onion_sweetpotato_kind_newval69": kind(on_sp(("newval",))),
            "onion_sweetpotato_state_dev48": [sum(a == y for y, a in on_sp(DEV)), len(on_sp(DEV))],
            "valid_kept": kept("valid"), "odev_kept": kept("odev"), "asian37_kept": kept("asian37"),
            "coco400_turned_away": [sum(a == AWAY for a in ans["coco400"]), len(ans["coco400"])],
            "rotten_fruit_turned_away_dev83": [sum(a == AWAY for _, a in rf), len(rf)],
            "five_dev_right_and_kept": [sum(a == y for y, a in pairs(DEV) if y in CLASS_NAMES[:10]), sum(y in CLASS_NAMES[:10] for y, _ in pairs(DEV))],
            "old_to_new": old, "answers": ans}


def main():
    sets = rt.photo_sets(); B, F = rt.outputs(sets)
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    g1 = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
    res = {"photos": {n: len(p) for n, (p, _) in sets.items()}, "TV-12": score(tv12, sets, B, F), "G1": score(g1, sets, B, F)}
    t, g = res["TV-12"], res["G1"]
    d = lambda k: g[k][0] - t[k][0]
    res["rules"] = {"green onion turned away >= 80%": g["green_onion_turned_away"][0] >= 0.8 * g["green_onion_turned_away"][1],
                    "green onion answered as another vegetable <= 6": sum(g["green_onion_answered_as"].values()) <= 6,
                    "onion / sweet potato kind, T3-T5 >= -2": d("onion_sweetpotato_kind_dev48") >= -2,
                    "onion / sweet potato kind, new-veg Valid >= -2": d("onion_sweetpotato_kind_newval69") >= -2,
                    "Valid kept >= -3": d("valid_kept") >= -3, "internet 221 kept >= -3": d("odev_kept") >= -3,
                    "Asian 37 kept >= -2": d("asian37_kept") >= -2, "everyday 400 turned away >= -5": d("coco400_turned_away") >= -5,
                    "rotten fruit turned away >= -3": d("rotten_fruit_turned_away_dev83") >= -3,
                    "old vegetables to new <= +1": g["old_to_new"] - t["old_to_new"] <= 1}
    res["passed"] = all(res["rules"].values())
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    for m in ("TV-12", "G1"):
        r = res[m]
        print(m, {k: v for k, v in r.items() if k != "answers"})
    print("passed:", res["passed"], {k: v for k, v in res["rules"].items() if not v})


if __name__ == "__main__":
    main()
