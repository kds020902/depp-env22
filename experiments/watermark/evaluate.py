"""Development checks of the watermark round (PREREG.md section 3); no blind test set is touched.

    python experiments/watermark/evaluate.py V7 WA W0 WB    -> dev_results.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from common import REPO, W, clip_features, dev_sets, system, valid_watermarks, vegetable_logits


def rotten(p: np.ndarray) -> np.ndarray:
    return p.argmax(1) % 2 == 1


def stamp_test(checkpoint, B, F, y) -> dict:
    fresh = y % 2 == 0
    base = system(checkpoint, B["valid"], F["valid"])
    out = {}
    for stamp in ("S1", "S2"):
        after = system(checkpoint, B[f"valid_{stamp}"], F[f"valid_{stamp}"])
        part = {}
        for key, name in (("five", "system"), ("vegetable_model", "vegetable model"), ("clip_head", "CLIP head")):
            b, a = rotten(base[key]), rotten(after[key])
            part[name] = {"fresh answered rotten": [int((b & fresh).sum()), int((a & fresh).sum())],
                          "rotten answered fresh": [int((~b & ~fresh).sum()), int((~a & ~fresh).sum())],
                          "fresh mean P(rotten)": [round(float(base[key][fresh][:, 1::2].sum(1).mean()), 3),
                                                   round(float(after[key][fresh][:, 1::2].sum(1).mean()), 3)]}
        part["increase"] = part["system"]["fresh answered rotten"][1] - part["system"]["fresh answered rotten"][0]
        out[stamp] = part
    return out


def scores(r, y) -> dict:
    right = r["index"] == y
    return {"accuracy": int(right.sum()), "kept_correct": int((right & ~r["unknown"]).sum()), "n": int(len(y)),
            "to_new_vegetable": int(((r["index"] >= 10) & ~r["unknown"] & (y < 10)).sum())}


def evaluate(tag: str) -> dict:
    sets = dev_sets(); F = clip_features(sets); B = vegetable_logits(W / f"{tag}.pth", tag, sets)
    checkpoint = torch.load(W / f"{tag}.pth", map_location="cpu", weights_only=True)
    res = {"tau": checkpoint["new_vegetable_head"]["switch_probability"]}
    y = sets["valid"][1]
    res["stamp_test"] = stamp_test(checkpoint, B, F, y)
    for name, (_, labels) in sets.items():
        r = system(checkpoint, B[name], F[name])
        if labels is None:
            res[name] = {"turned_away": int(r["unknown"].sum()), "n": len(r["unknown"]),
                         "to_new_vegetable": int((r["switched"] & ~r["unknown"]).sum())}
        elif name == "newval":
            res[name] = {"kind_right_kept": int(((r["index"] // 2 == labels // 2) & ~r["unknown"]).sum()), "n": int(len(labels))}
        else:
            res[name] = scores(r, labels)
    r = system(checkpoint, B["valid"], F["valid"]); wm = valid_watermarks().astype(bool); fresh = y % 2 == 0
    pick = lambda m: {k: v[m] for k, v in r.items()}
    res["valid_no_watermark"] = scores(pick(~wm), y[~wm])
    res["valid_watermark"] = scores(pick(wm), y[wm])
    res["valid_fresh_with_watermark"] = scores(pick(wm & fresh), y[wm & fresh])
    res["valid_rotten_without_watermark"] = scores(pick(~wm & ~fresh), y[~wm & ~fresh])
    for key, name in (("vegetable_model", "vegetable_model_alone"), ("clip_head", "clip_head_alone")):
        res[name] = {"valid": int((r[key].argmax(1) == y).sum()), "valid_no_watermark": int((r[key][~wm].argmax(1) == y[~wm]).sum())}
    return res


def rules(res: dict, base: dict) -> dict:
    d = lambda k, f: res[k][f] - base[k][f]
    checks = {
        "D1 S1 increase <= 3": res["stamp_test"]["S1"]["increase"] <= 3,
        "D1 S2 increase <= 3": res["stamp_test"]["S2"]["increase"] <= 3,
        "D2 Valid kept >= V7-3": d("valid", "kept_correct") >= -3,
        "D2 internet 221 accuracy >= V7-3": d("odev", "accuracy") >= -3,
        "D2 internet 221 kept >= V7-3": d("odev", "kept_correct") >= -3,
        "D2 160 kept >= V7-3": d("o160", "kept_correct") >= -3,
        "D2 Asian 37 kept >= V7-2": d("asian37", "kept_correct") >= -2,
        "D2 fruit 374 turned away >= V7-10": d("fruit374", "turned_away") >= -10,
        "D2 everyday 400 turned away >= V7-5": d("coco400", "turned_away") >= -5,
        "D2 new-vegetable Valid >= V7-2": d("newval", "kind_right_kept") >= -2,
        "D3 Valid without watermark kept >= V7-2": d("valid_no_watermark", "kept_correct") >= -2,
    }
    return {"checks": checks, "pass": all(checks.values()),
            "S1+S2 increase": res["stamp_test"]["S1"]["increase"] + res["stamp_test"]["S2"]["increase"]}


def main():
    tags = sys.argv[1:] or ["V7", "WA", "W0", "WB"]
    path = REPO / "experiments/watermark/dev_results.json"
    out = json.load(open(path)) if path.exists() else {}
    for tag in tags:
        out[tag] = evaluate(tag)
        print(tag, json.dumps(out[tag], ensure_ascii=False), flush=True)
    for tag in [t for t in out if t not in ("V7", "rules") and not t.startswith("chosen")]:
        if "V7" in out:
            out.setdefault("rules", {})[tag] = rules(out[tag], out["V7"])
    # round 1: WA, WB (ties -> WA); round 2 (PREREG.md, light watermarks): WC, WD (ties -> WD, the smaller change)
    for key, pair, simpler in (("chosen", ("WA", "WB"), "WA"), ("chosen_round2", ("WC", "WD"), "WD")):
        candidates = {t: r for t, r in out.get("rules", {}).items() if t in pair and r["pass"]}
        order = sorted(candidates, key=lambda t: (candidates[t]["S1+S2 increase"],
                                                  -out[t]["valid_no_watermark"]["kept_correct"], t != simpler))
        out[key] = order[0] if order else None
    json.dump(out, open(path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({"rules": out.get("rules"), "chosen": out["chosen"], "chosen_round2": out["chosen_round2"]},
                     indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
