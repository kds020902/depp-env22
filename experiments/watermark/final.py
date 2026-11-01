"""Final scoring of the watermark round (PREREG.md section 4): the candidate chosen on development photos
(dev_results.json "chosen") and V7 (adopted now), scored once on Test 300, the new internet test 241, the any-photo set v2
and new-vegetable tests 1-4. V7's numbers must come out as recorded before (pipeline check).

    python experiments/watermark/final.py    -> final_once.json
"""

from __future__ import annotations

import collections
import csv
import json

import numpy as np
import torch

from common import REPO, S, W, system
from model_utils import CLASS_NAMES, input_size, load_model, make_transform, open_image
from clip_model import ClipModel
from stamps import STAMPS

REJECT = ("rotten_fruit", "fresh_fruit", "other_veg", "food", "drawing_toy", "everyday")
NEW = ("onion", "sweetpotato", "greenonion")


def photo_sets():
    test = [r for r in csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")) if r["split"] == "test"]
    sets = {"test_300": ([str(REPO / "dataset" / r["image_path"]) for r in test], [r["class_name"] for r in test])}
    t2 = list(csv.DictReader(open(REPO / "results/10class/outside_test2/outside_test2.csv", encoding="utf-8")))
    sets["new_test_241"] = ([str(S / f"weak/t2img/{r['id']}.jpg") for r in t2], [r["class_name"] for r in t2])
    rows = list(csv.DictReader(open(S / "arb/anyphoto_test_v2.csv", encoding="utf-8"))); files = json.load(open(S / "arb/local_files.json"))
    sets["any_photo_v2"] = ([files[r["id"]] for r in rows], rows)
    pool = json.load(open(S / "v7/pool.json"))
    for k, rnd in enumerate(("v4", "v5", "v6", "v7"), 1):
        te = [r for r in pool if r["round"] == rnd and r["split"] == "test" and r["class_name"].startswith(NEW)]
        sets[f"new_veg_test{k}"] = ([r["path"] for r in te], [r["class_name"] for r in te])
    return sets


def features(sets):
    cache = W / "final_clip.npz"
    have = dict(np.load(cache)) if cache.exists() else {}
    clip = None
    jobs = [(n, n, None) for n in sets] + [(f"test_300_{s}", "test_300", s) for s in STAMPS]
    for key, name, stamp in jobs:
        if key in have:
            continue
        clip = clip or ClipModel("cpu")
        out = []
        for i in range(0, len(sets[name][0]), 32):
            ims = [open_image(p) for p in sets[name][0][i:i + 32]]
            out.append(clip.image_features([STAMPS[stamp](im) for im in ims] if stamp else ims).cpu())
        have[key] = torch.cat(out).numpy(); np.savez(cache, **have)
    return have


def logits(tag, sets):
    cache = W / f"final_b_{tag}.npz"
    have = dict(np.load(cache)) if cache.exists() else {}
    model, checkpoint = load_model(W / f"{tag}.pth", torch.device("cpu")); tf = make_transform(input_size(checkpoint))
    jobs = [(n, n, None) for n in sets] + [(f"test_300_{s}", "test_300", s) for s in STAMPS]
    for key, name, stamp in jobs:
        if key in have:
            continue
        out = []
        with torch.inference_mode():
            for i in range(0, len(sets[name][0]), 32):
                ims = [open_image(p) for p in sets[name][0][i:i + 32]]
                out.append(model(torch.stack([tf(STAMPS[stamp](im) if stamp else im) for im in ims])))
        have[key] = torch.cat(out).numpy(); np.savez(cache, **have)
    return have


def veg_scores(r, y, mask=None):
    m = np.ones(len(y), bool) if mask is None else mask
    right = (r["index"] == y) & m; kept = right & ~r["unknown"]
    return {"n": int(m.sum()), "accuracy": int(right.sum()), "kept_correct": int(kept.sum()), "turned_away": int((r["unknown"] & m).sum()),
            "answered_new_vegetable": int(((r["index"] >= 10) & ~r["unknown"] & m).sum()),
            "by_class": {n: f"{int(right[(y == i) & m].sum())}/{int(((y == i) & m).sum())}" for i, n in enumerate(CLASS_NAMES) if ((y == i) & m).any()}}


def any_photo(r, rows):
    good, cats = [], []
    for row, p, j in zip(rows, r["index"], r["unknown"]):
        c = row["category"]; lab = row["label"].split("|") if row["label"] else []; pred = CLASS_NAMES[p]
        if c == "excluded_v2":
            continue
        if c in REJECT: g = bool(j)
        elif c == "five_daily": g = (not j) and pred == lab[0]
        elif c == "new_veg": g = (not j) and (pred == lab[0] or (lab[0].startswith("greenonion") and pred.startswith("greenonion")))
        elif c == "cut_five": g = bool(j) or pred == lab[0]
        else: g = bool(j) or pred in lab or (pred.startswith("greenonion") and any(l.startswith("greenonion") for l in lab))
        good.append(g); cats.append(c)
    good, cats = np.array(good), np.array(cats); old = cats != "new_veg"
    return {"good_old_categories": int(good[old].sum()), "of_old": int(old.sum()), "good_new_veg": int(good[~old].sum()),
            "of_new_veg": int((~old).sum()), "by_category": {c: f"{int(good[cats == c].sum())}/{int((cats == c).sum())}" for c in dict.fromkeys(cats)}}


def score(tag, sets, F):
    B = logits(tag, sets); checkpoint = torch.load(W / f"{tag}.pth", map_location="cpu", weights_only=True)
    out = {}
    wm = {r["image_path"]: int(r["watermark"]) for r in csv.DictReader(open(REPO / "experiments/watermark/valid_test_watermarks.csv"))}
    for name, (paths, labels) in sets.items():
        r = system(checkpoint, B[name], F[name])
        if name == "any_photo_v2":
            out[name] = any_photo(r, labels); continue
        y = np.array([CLASS_NAMES.index(c) for c in labels])
        if name.startswith("new_veg_test"):
            kind = (r["index"] // 2 == y // 2) & ~r["unknown"]
            out[name] = {"n": len(y), "kind_right_and_kept": int(kind.sum()), "rejected": int(r["unknown"].sum()),
                         "misses": collections.Counter(f'{CLASS_NAMES[a]}->{"REJ" if j else CLASS_NAMES[q]}' for a, q, j in
                                                       zip(y, r["index"], r["unknown"]) if not (q // 2 == a // 2 and not j)).most_common()}
            continue
        out[name] = veg_scores(r, y)
        if name == "test_300":
            w = np.array([wm[p.split("/dataset/", 1)[1]] for p in paths]).astype(bool)
            out["test_no_watermark"] = veg_scores(r, y, ~w); out["test_watermark"] = veg_scores(r, y, w)
            fresh = y % 2 == 0
            for s in STAMPS:
                a = system(checkpoint, B[f"test_300_{s}"], F[f"test_300_{s}"])
                out[f"test_fresh_{s}"] = {"fresh answered rotten": [int(((r["five"].argmax(1) % 2 == 1) & fresh).sum()),
                                                                    int(((a["five"].argmax(1) % 2 == 1) & fresh).sum())]}
    out["new_veg_tests_total"] = sum(out[f"new_veg_test{k}"]["kind_right_and_kept"] for k in range(1, 5))
    return out


def main():
    dev = json.load(open(REPO / "experiments/watermark/dev_results.json"))
    tag = dev["chosen"]; assert tag in ("WA", "WB"), tag
    sets = photo_sets(); F = features(sets)
    res = {"candidate": tag, "V7": score("V7", sets, F)}
    v7 = res["V7"]
    assert (v7["test_300"]["accuracy"], v7["test_300"]["kept_correct"]) == (288, 285), v7["test_300"]
    assert (v7["new_test_241"]["accuracy"], v7["new_test_241"]["kept_correct"]) == (200, 190), v7["new_test_241"]
    assert v7["any_photo_v2"]["good_old_categories"] == 203 and v7["new_veg_tests_total"] == 176, (v7["any_photo_v2"], v7["new_veg_tests_total"])
    res[tag] = c = score(tag, sets, F)
    rules = {
        "Test accuracy >= 284": c["test_300"]["accuracy"] >= 284,
        "Test kept >= 281": c["test_300"]["kept_correct"] >= 281,
        "Test without watermark kept >= V7-2": c["test_no_watermark"]["kept_correct"] >= v7["test_no_watermark"]["kept_correct"] - 2,
        "internet 241 accuracy >= 199": c["new_test_241"]["accuracy"] >= 199,
        "internet 241 kept >= 189": c["new_test_241"]["kept_correct"] >= 189,
        "any-photo old 247 good >= 200": c["any_photo_v2"]["good_old_categories"] >= 200,
        "any-photo new 6 good >= 5": c["any_photo_v2"]["good_new_veg"] >= 5,
        "new-vegetable tests 1-4 >= 173": c["new_veg_tests_total"] >= 173,
    }
    res["rules"] = rules; res["adopt"] = all(rules.values())
    json.dump(res, open(REPO / "experiments/watermark/final_once.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
