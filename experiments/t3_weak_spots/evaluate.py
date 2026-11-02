"""Development check of PREREG.md: TV-12 against the candidates on the development photos only (T4 is not touched).

    python experiments/t3_weak_spots/evaluate.py K1 [K2]   -> dev_results.json

Uses the cached CLIP features / vegetable-model logits of the earlier development sets (record script) and the
development share of the newly reviewed photos (t4/lists/dev.csv in the working folder).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "watermark"))
from common import OLD_DEV, REPO, S, W, clip_features, dev_sets, system, vegetable_logits  # noqa: E402
from model_utils import CLASS_NAMES, input_size, load_model, make_transform, open_image  # noqa: E402

LISTS = Path(os.environ.get("T3WS_LISTS", S / "t4/lists"))  # round 2: t5/lists
OUT = os.environ.get("T3WS_OUT", "dev_results.json")
WEAK = ("potato_fresh", "carrot_fresh", "tomato_rotten", "sweetpotato_fresh", "rotten_fruit")
PAIR = ("potato_rotten", "tomato_fresh", "carrot_rotten")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


zs = _load("zs_evaluate", REPO / "experiments/zero_shot_reject/evaluate.py")


def new_dev(tag: str, path: Path):
    """CLIP features (cached once) and vegetable-model logits (cached per candidate) of the development share."""
    rows = list(csv.DictReader(open(LISTS / "dev.csv", encoding="utf-8")))
    sfx = "" if LISTS.parent.name == "t4" else "_" + LISTS.parent.name
    fcache, bcache = W / f"t3ws_dev_clip{sfx}.npz", W / f"t3ws_dev_b_{tag}{sfx}.npz"
    paths = [r["path"] for r in rows]
    if fcache.exists():
        F = np.load(fcache)["f"]
    else:
        from clip_model import ClipModel
        clip = ClipModel("cpu")
        with torch.inference_mode():
            F = torch.cat([clip.image_features([open_image(p) for p in paths[i:i + 32]]).cpu() for i in range(0, len(paths), 32)]).numpy()
        np.savez(fcache, f=F)
    if bcache.exists():
        B = np.load(bcache)["b"]
    else:
        model, ck = load_model(path, torch.device("cpu")); tf = make_transform(input_size(ck))
        with torch.inference_mode():
            B = torch.cat([model(torch.stack([tf(open_image(p)) for p in paths[i:i + 32]])) for i in range(0, len(paths), 32)]).numpy()
        np.savez(bcache, b=B)
    return rows, F, B


def score(tag: str, path: Path, b_tag: str) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sets = dev_sets(); F = clip_features(sets)
    B = vegetable_logits(path, b_tag, sets) if b_tag != "WC" else dict(np.load(W / "b_WC.npz"))
    out = {}
    for name in ("valid", "odev", "o160", "asian37", "coco400", "lvis", "newval"):
        r = system(ck, B[name], F[name]); y = sets[name][1]
        if y is None:
            out[name] = {"turned_away": int(r["unknown"].sum()), "to_new_vegetable": int(((r["index"] >= 10) & ~r["unknown"]).sum())}
        elif name == "newval":
            out[name] = {"kind_right_kept": int(((r["index"] // 2 == y // 2) & ~r["unknown"]).sum())}
        else:
            out[name] = {"kept_correct": int(((r["index"] == y) & ~r["unknown"]).sum()),
                         "to_new_vegetable": int(((r["index"] >= 10) & ~r["unknown"]).sum())}
    out["old_dev_to_new"] = sum(out[n]["to_new_vegetable"] for n in OLD_DEV)
    rf = zs.rotten_fruit_features(); X = np.concatenate(list(rf.values()))
    out["old_rotten_fruit_761_turned_away"] = int(system(ck, np.zeros((len(X), 10), np.float32), X)["unknown"].sum())
    rows, Fn, Bn = new_dev(b_tag, path)
    r = system(ck, Bn, Fn); per = {}
    for c in sorted({x["class_name"] for x in rows}):
        m = np.array([x["class_name"] == c for x in rows])
        if c == "rotten_fruit":
            good = r["unknown"][m]
        elif c.startswith("sweetpotato"):
            good = (r["index"][m] // 2 == CLASS_NAMES.index(c) // 2) & ~r["unknown"][m]
        else:
            good = (r["index"][m] == CLASS_NAMES.index(c)) & ~r["unknown"][m]
        per[c] = [int(good.sum()), int(m.sum())]
    out["new_dev"] = per
    out["weak_total"] = sum(per.get(c, [0])[0] for c in WEAK)
    out["pair_total"] = sum(per.get(c, [0])[0] for c in PAIR)
    return out


def dev_rules(c: dict, base: dict) -> dict:
    """The 8 development rules (PREREG.md): what a candidate (score() output) must keep compared with the base model."""
    d = lambda n, f: c[n][f] - base[n][f]
    return {"Valid kept >= -3": d("valid", "kept_correct") >= -3, "internet 221 kept >= -3": d("odev", "kept_correct") >= -3,
            "160 kept >= -3": d("o160", "kept_correct") >= -3, "Asian 37 kept >= -2": d("asian37", "kept_correct") >= -2,
            "everyday 400 turned away >= -5": d("coco400", "turned_away") >= -5,
            "new-vegetable Valid >= -2": d("newval", "kind_right_kept") >= -2,
            "old-vegetable dev to new <= +1": c["old_dev_to_new"] - base["old_dev_to_new"] <= 1,
            "pair classes >= -2": c["pair_total"] - base["pair_total"] >= -2}


def main():
    res = {"TV-12": score("TV-12", REPO / "model.pth", "WC")}
    for tag in sys.argv[1:]:
        res[tag] = score(tag, W / f"{tag}.pth", "WC" if tag in ("K1", "K3", "L1") else ("K2" if tag in ("K2", "K4") else tag))
    base = res["TV-12"]; rules = {}
    for tag in sys.argv[1:]:
        checks = dev_rules(res[tag], base)
        rules[tag] = {"checks": checks, "pass": all(checks.values()), "weak_gain": res[tag]["weak_total"] - base["weak_total"]}
    ok = [t for t in rules if rules[t]["pass"]]
    best = max(ok, key=lambda t: rules[t]["weak_gain"]) if ok else None
    chosen = best if best is not None and rules[best]["weak_gain"] >= 10 else None
    res.update(rules=rules, chosen=chosen)
    json.dump(res, open(HERE / OUT, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
