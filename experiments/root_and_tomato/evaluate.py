"""experiments/root_and_tomato (PREREG.md): TV-12, the rebuilt base B0' and the candidate on the development photos.

    python experiments/root_and_tomato/evaluate.py OUT.json B0p=$DEPP_WORK/wm/B0p.pth R1=$DEPP_WORK/wm/R1.pth

Development photos: T3, T4 and T5 (all used before, so development now). The keep-rule sets (Valid, internet 221, Asian 37,
everyday 400, new-vegetable Valid, LVIS crops; also the field 160 and fruit 374 for reference) use the CLIP features and
vegetable-model outputs stored in experiments/new_vegetables/v7/head_data.npz. The vegetable model is the same in every
system (model.pth). No blind test set is read.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from clip_model import ClipModel  # noqa: E402
from decision import decide  # noqa: E402
from images import make_transform, open_image  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402
from networks import input_size, load_model  # noqa: E402

S = Path(os.environ["DEPP_WORK"])
CACHE = S / "wm/root_and_tomato_cache.npz"
AWAY = "거름"
REJECT = ("rotten_fruit", "fresh_fruit", "other_veg")
NEW = ("onion", "sweetpotato", "greenonion")


def photo_sets() -> dict[str, tuple[list[str], list[str | None]]]:
    """T3, T4, T5: name -> (paths, labels), only photos on disk (the same photos for every model)."""
    out = {}
    for t, folder in (("t3", S / "t3/img"), ("t4", S / "t4/img"), ("t5", S / "t5test/img")):
        rows = list(csv.DictReader(open(REPO / f"results/{t}_test/{t}_test.csv", encoding="utf-8")))
        keep = [r for r in rows if (folder / f"{r['id']}.jpg").exists()]
        out[t] = ([str(folder / f"{r['id']}.jpg") for r in keep], [r["class_name"] for r in keep])
    return out


def outputs(sets) -> tuple[dict, dict]:
    """Vegetable-model logits and CLIP features: computed for T3/T4/T5 (cached); the keep-rule sets come from the stored
    features of experiments/new_vegetables/v7/head_data.npz (the same image model and model.pth, so no download is needed)."""
    have = dict(np.load(CACHE)) if CACHE.exists() else {}
    model = clip = None
    for name, (paths, _) in sets.items():
        key = f"{name}:{len(paths)}"
        if f"b|{key}" in have:
            continue
        if model is None:
            model, ck = load_model(REPO / "model.pth", torch.device("cpu")); tf = make_transform(input_size(ck)); clip = ClipModel("cpu")
        B, F = [], []
        with torch.inference_mode():
            for i in range(0, len(paths), 32):
                ims = [open_image(p) for p in paths[i:i + 32]]
                B.append(model(torch.stack([tf(im) for im in ims]))); F.append(clip.image_features(ims).cpu())
        have[f"b|{key}"], have[f"f|{key}"] = torch.cat(B).numpy(), torch.cat(F).numpy()
        np.savez(CACHE, **have)
        print("computed", key, flush=True)
    B = {n: have[f"b|{n}:{len(p)}"] for n, (p, _) in sets.items()}; F = {n: have[f"f|{n}:{len(p)}"] for n, (p, _) in sets.items()}
    stored = np.load(REPO / "experiments/new_vegetables/v7/head_data.npz")
    for n in ("valid", "odev", "o160", "asian37", "fruit374", "coco400", "lvis", "newval"):
        B[n], F[n] = stored[f"dev_{n}_b"], stored[f"dev_{n}_X"]
        y = stored[f"dev_{n}_y"] if f"dev_{n}_y" in stored.files else None
        sets[n] = ([f"{n}#{i}" for i in range(len(B[n]))], [CLASS_NAMES[int(v)] for v in y] if y is not None else [None] * len(B[n]))
    return B, F


def answers(ck: dict, b: np.ndarray, f: np.ndarray) -> list[str]:
    a = decide(torch.tensor(b), ck, torch.tensor(f))
    return [AWAY if u else CLASS_NAMES[int(i)] for i, u in zip(a.index, a.unknown)]


def good(y: str | None, a: str) -> bool:
    if y is None or y in REJECT:
        return a == AWAY
    if y.startswith(NEW):
        return a != AWAY and a.rsplit("_", 1)[0] == y.rsplit("_", 1)[0]
    return a == y


def score(ck: dict, sets, B, F) -> dict:
    ans = {n: answers(ck, B[n], F[n]) for n in sets}
    per = {}
    for t in ("t3", "t4", "t5"):
        for y, a in zip(sets[t][1], ans[t]):
            per.setdefault(y, [0, 0]); per[y][0] += good(y, a); per[y][1] += 1
    state = {y: sum(a == y for t in ("t3", "t4", "t5") for yy, a in zip(sets[t][1], ans[t]) if yy == y) for y in per if y.startswith(("onion", "sweetpotato"))}
    old = lambda n: sum(a != AWAY and CLASS_NAMES.index(a) >= 10 for a in ans[n])
    kept = lambda n: sum(a == y for y, a in zip(sets[n][1], ans[n]))
    five_dev = [(y, a) for t in ("t3", "t4", "t5") for y, a in zip(sets[t][1], ans[t]) if y in CLASS_NAMES[:10]]
    return {"dev_per_class": per, "dev_state_right_new_veg": state,
            "dev_five_right_and_kept": [sum(y == a for y, a in five_dev), len(five_dev)],
            "dev_five_rotten_to_fresh": sum(y.endswith("_rotten") and a == y.replace("_rotten", "_fresh") for y, a in five_dev),
            "new_veg_kind_61": [sum(per[c][0] for c in per if c.startswith(NEW)), sum(per[c][1] for c in per if c.startswith(NEW))],
            "valid_kept": [kept("valid"), len(sets["valid"][0])], "odev_kept": [kept("odev"), len(sets["odev"][0])],
            "asian37_kept": [kept("asian37"), len(sets["asian37"][0])],
            "coco400_turned_away": [sum(a == AWAY for a in ans["coco400"]), len(sets["coco400"][0])],
            "fruit374_turned_away": [sum(a == AWAY for a in ans["fruit374"]), len(sets["fruit374"][0])],
            "o160_kept_reference_only": [kept("o160"), len(sets["o160"][0])],
            "newval_kind": [sum(good(y, a) for y, a in zip(sets["newval"][1], ans["newval"])), len(sets["newval"][0])],
            "old_to_new": sum(old(n) for n in ("valid", "odev", "asian37", "lvis")), "answers": ans}


def rules(c: dict, base: dict) -> dict:
    """PREREG.md: development bars and the seven keep rules, candidate against TV-12."""
    d = lambda cls: c["dev_per_class"][cls][0] - base["dev_per_class"][cls][0]
    k = lambda key, i=0: c[key][i] - base[key][i]
    return {"fresh potato +5 (bar)": d("potato_fresh") >= 5, "rotten tomato +3 (bar)": d("tomato_rotten") >= 3,
            "rotten potato >= -2": d("potato_rotten") >= -2, "fresh tomato >= -2": d("tomato_fresh") >= -2,
            "onion / sweet potato / green onion kind >= -2": k("new_veg_kind_61") >= -2, "rotten fruit turned away >= -3": d("rotten_fruit") >= -3,
            "Valid kept >= -3": k("valid_kept") >= -3, "internet 221 kept >= -3": k("odev_kept") >= -3, "Asian 37 kept >= -2": k("asian37_kept") >= -2,
            "everyday 400 turned away >= -5": k("coco400_turned_away") >= -5, "new-vegetable Valid >= -2": k("newval_kind") >= -2,
            "old vegetables to new <= +1": c["old_to_new"] - base["old_to_new"] <= 1}


def main():
    out_path, cands = Path(sys.argv[1]), dict(a.split("=", 1) for a in sys.argv[2:])
    sets = photo_sets(); B, F = outputs(sets)
    res = {"photos": {n: len(p) for n, (p, _) in sets.items()}}
    models = {"TV-12": REPO / "model.pth", **{k: Path(v) for k, v in cands.items()}}
    for name, path in models.items():
        res[name] = score(torch.load(path, map_location="cpu", weights_only=True), sets, B, F)
    for name in cands:
        r = rules(res[name], res["TV-12"]); bars = r["fresh potato +5 (bar)"] or r["rotten tomato +3 (bar)"]
        keep = all(v for k, v in r.items() if "(bar)" not in k)
        res[f"{name}_rules"] = r; res[f"{name}_to_T6"] = bool(bars and keep)
        res[f"{name}_same_answers_as_TV-12"] = {n: sum(x == y for x, y in zip(res[name]["answers"][n], res["TV-12"]["answers"][n])) for n in sets}
    json.dump(res, open(out_path, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res["photos"]))
    for name in models:
        r = res[name]; p = r["dev_per_class"]
        print(f"{name:6} fresh potato {p['potato_fresh']} rotten potato {p['potato_rotten']} rotten tomato {p['tomato_rotten']} fresh tomato {p['tomato_fresh']} "
              f"new-veg kind {r['new_veg_kind_61']} rotten fruit {p['rotten_fruit']} | valid {r['valid_kept']} odev {r['odev_kept']} asian {r['asian37_kept']} "
              f"coco {r['coco400_turned_away']} newval {r['newval_kind']} old->new {r['old_to_new']}")
    for name in cands:
        print(name, "to T6:", res[f"{name}_to_T6"], {k: v for k, v in res[f"{name}_rules"].items() if not v})


if __name__ == "__main__":
    main()
