"""experiments/dirty_potato (PREREG.md): TV-12 against P1 (t3_weak_spots L1) on the development photos, T3 and T4 included
(both used twice, so development now). No blind test set is read.

    python experiments/dirty_potato/evaluate.py OUT.json
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/watermark"))
from common import S, W, dev_rules  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eye = _load("eye_evaluate", REPO / "experiments/bigger_eye/evaluate.py")
ref = _load("eye_t3_t4", REPO / "experiments/bigger_eye/t3_t4_reference.py")
POTATO = ("potato_fresh", "potato_rotten")


def per_class(rows: list[dict], answers: list[str]) -> dict:
    out = {}
    for c in dict.fromkeys(r["class_name"] for r in rows):
        idx = [i for i, r in enumerate(rows) if r["class_name"] == c]
        want = ref.TURNED_AWAY if c in ref.REJECT else c
        if c.startswith(("onion", "sweetpotato", "greenonion")):
            good = sum(answers[i] != ref.TURNED_AWAY and answers[i].rsplit("_", 1)[0] == c.rsplit("_", 1)[0] for i in idx)
        else:
            good = sum(answers[i] == want for i in idx)
        out[c] = [good, len(idx)]
    return out


def main():
    lists = json.load(open(W / "eye_lists.json"))
    feats = eye.load("B0")                               # TV-12's own image model (ViT-B-16); P1 uses the same one
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    p1 = torch.load(W / "L1.pth", map_location="cpu", weights_only=True)
    res = {"TV-12": {}, "P1": {}}
    res["TV-12"]["dev"], _ = eye.score(tv12, feats, lists)
    res["P1"]["dev"], _ = eye.score(p1, feats, lists)
    res["P1"]["rules"] = dev_rules(res["P1"]["dev"], res["TV-12"]["dev"])
    # T3 and T4 as development photos: the same features and vegetable-model outputs for both systems
    cache = W / "dirty_potato_t3t4.npz"
    stored = {"t3": json.load(open(REPO / "results/t3_test/scores_once.json"))["ZS8"]["answers"],
              "t4": json.load(open(REPO / "results/t4_test/scores_reference.json"))["TV-12"]["answers"]}
    rows = {"t3": list(csv.DictReader(open(REPO / "results/t3_test/t3_test.csv", encoding="utf-8"))),
            "t4": list(csv.DictReader(open(REPO / "results/t4_test/t4_test.csv", encoding="utf-8")))}
    if cache.exists():
        z = dict(np.load(cache))
    else:
        from clip_model import ClipModel
        clip = {"TV-12": ClipModel("cpu")}; z = {}
        for s, img in (("t3", S / "t3/img"), ("t4", S / "t4/img")):
            b, F = ref.outputs([str(img / f"{r['id']}.jpg") for r in rows[s]], clip)
            z[f"{s}_b"], z[f"{s}_f"] = b, F["TV-12"]
        np.savez(cache, **z)
    for s in ("t3", "t4"):
        a12 = ref.names(tv12, z[f"{s}_b"], z[f"{s}_f"])
        res["TV-12"][f"{s}_pipeline_differences"] = sum(x != y for x, y in zip(a12, stored[s]))
        res["TV-12"][s] = per_class(rows[s], a12)
        res["P1"][s] = per_class(rows[s], ref.names(p1, z[f"{s}_b"], z[f"{s}_f"]))
    for m in res:
        d = res[m]["dev"]["new_dev"]
        res[m]["potato_fresh_60"] = d["potato_fresh"][0] + res[m]["t3"]["potato_fresh"][0] + res[m]["t4"]["potato_fresh"][0]
        res[m]["potato_rotten_36"] = d["potato_rotten"][0] + res[m]["t3"]["potato_rotten"][0] + res[m]["t4"]["potato_rotten"][0]
        res[m]["tomato_rotten_42"] = d["tomato_rotten"][0] + res[m]["t3"]["tomato_rotten"][0] + res[m]["t4"]["tomato_rotten"][0]
    t, p = res["TV-12"], res["P1"]
    res["verdict"] = {"fresh potato gain (>= +6)": p["potato_fresh_60"] - t["potato_fresh_60"],
                      "rotten potato change (>= -2)": p["potato_rotten_36"] - t["potato_rotten_36"],
                      "8 guard rules": all(p["rules"].values())}
    res["verdict"]["improvement"] = (res["verdict"]["fresh potato gain (>= +6)"] >= 6 and res["verdict"]["rotten potato change (>= -2)"] >= -2
                                     and res["verdict"]["8 guard rules"])
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    for m in res:
        if m == "verdict":
            continue
        r = res[m]; o = r["dev"]["outside"]
        print(m, "dev", r["dev"]["dev_total"], "| outside answered", o["answered"], "conf-wrong", o["confident_wrong"], "rot->fresh", o["rotten_to_fresh_answered"],
              "| rotfruit761", r["dev"]["old_rotten_fruit_761_turned_away"], "| fresh potato 60:", r["potato_fresh_60"], "rotten potato 36:", r["potato_rotten_36"],
              "rotten tomato 42:", r["tomato_rotten_42"])
        print("   T3", r["t3"]); print("   T4", r["t4"])
    print("pipeline check (TV-12 vs stored):", t["t3_pipeline_differences"], t["t4_pipeline_differences"])
    print(json.dumps(res["verdict"], ensure_ascii=False), p["rules"])


if __name__ == "__main__":
    main()
