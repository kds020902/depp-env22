"""experiments/bigger_eye (PREREG.md): development check of the candidates made by build.py, and the pre-registered choice.

    python experiments/bigger_eye/evaluate.py OUT.json B0 B1 B2 [L1]

TV-12 itself is scored with model.pth on B0's features (TV-12's own image model). Every candidate uses the same vegetable-model
outputs (TV-12's EfficientNet-B0) and its own image features. No blind test set is read here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "watermark"))
from common import OLD_DEV, PAIR, REPO, WEAK, W, dev_rules  # noqa: E402
from decision import decide, held_back  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402

LABELLED = ("valid", "odev", "o160", "asian37")
OUTSIDE = ("odev", "o160", "asian37")
WEB = {"B0", "B1", "B2"}   # image models small enough for the browser demo


def load(tag: str):
    with np.load(W / f"eye_{tag}_feats.npz") as saved:
        store = dict(zip(saved["keys"].tolist(), saved["features"]))
    return lambda paths: torch.tensor(np.stack([store[p] for p in paths]))


def score(ck: dict, feats, lists: dict, veg: dict | None = None) -> dict:
    """veg: the vegetable-model outputs to use ({"dev": {set: logits}, "t5_dev": logits, "test_300": logits});
    default TV-12's (WC), cached at their last scoring."""
    veg = veg or {"dev": dict(np.load(W / "b_WC.npz")), "t5_dev": np.load(W / "t3ws_dev_b_WC_t5.npz")["b"],
                  "test_300": np.load(W / "final_b_WC.npz")["test_300"]}
    B = veg["dev"]; dev = lists["dev"]
    sets = json.load(open(Path(W).parent / "clip/sets.json"))
    labels = {n: np.array([CLASS_NAMES.index(c) for _, c in sets[n]]) for n in LABELLED}
    pool = json.load(open(Path(W).parent / "v7/pool.json"))
    labels["newval"] = np.array([CLASS_NAMES.index(r["class_name"]) for r in pool if r["split"] == "val"])
    out, answers = {}, {}
    for name in ("valid", "odev", "o160", "asian37", "fruit374", "coco400", "lvis", "newval"):
        a = decide(torch.tensor(B[name]), ck, feats(dev[name])); answers[name] = a
        index, unknown = a.index.numpy(), a.unknown.numpy(); to_new = int(((index >= 10) & ~unknown).sum())
        if name in LABELLED:
            y = labels[name]; right = (index == y) & ~unknown; held = held_back(a).numpy(); answered = ~unknown & ~held
            out[name] = {"kept_correct": int(right.sum()), "to_new_vegetable": to_new, "known": int((~unknown).sum()),
                         "answered": int(answered.sum()), "answered_right": int((answered & right).sum()),
                         "confident_wrong": int((answered & ~right).sum()),
                         "rotten_to_fresh_answered": int((answered & (y % 2 == 1) & (index < 10) & (index % 2 == 0)).sum())}
        elif name == "newval":
            out[name] = {"kind_right_kept": int(((index // 2 == labels[name] // 2) & ~unknown).sum())}
        else:
            out[name] = {"turned_away": int(unknown.sum()), "to_new_vegetable": to_new}
    out["old_dev_to_new"] = sum(out[n]["to_new_vegetable"] for n in OLD_DEV)
    X = torch.cat([feats(v) for v in lists["rotten_fruit"].values()])
    out["old_rotten_fruit_761_turned_away"] = int(decide(torch.zeros(len(X), 10), ck, X).unknown.sum())
    # weak-spot development share (experiments/t3_weak_spots, round 2)
    import csv
    from common import DEV_LISTS
    rows = list(csv.DictReader(open(DEV_LISTS / "dev.csv", encoding="utf-8")))
    a = decide(torch.tensor(veg["t5_dev"]), ck, feats(lists["t5_dev"]))
    index, unknown, per = a.index.numpy(), a.unknown.numpy(), {}
    for c in sorted({r["class_name"] for r in rows}):
        m = np.array([r["class_name"] == c for r in rows])
        if c == "rotten_fruit":
            good = unknown[m]
        elif c.startswith("sweetpotato"):
            good = (index[m] // 2 == CLASS_NAMES.index(c) // 2) & ~unknown[m]
        else:
            good = (index[m] == CLASS_NAMES.index(c)) & ~unknown[m]
        per[c] = [int(good.sum()), int(m.sum())]
    out["new_dev"] = per
    out["weak_total"] = sum(per.get(c, [0])[0] for c in WEAK)
    out["pair_total"] = sum(per.get(c, [0])[0] for c in PAIR)
    out["outside"] = {k: sum(out[n][k] for n in OUTSIDE) for k in out["odev"]}
    out["dev_total"] = out["valid"]["kept_correct"] + out["outside"]["kept_correct"]
    # Test 300: not blind any more (its errors were looked at), reference only
    t = decide(torch.tensor(veg["test_300"]), ck, feats(lists["test_300"]))
    import csv as _csv
    yt = np.array([CLASS_NAMES.index(r["class_name"]) for r in _csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8"))
                   if r["split"] == "test"])
    right = (t.index.numpy() == yt) & ~t.unknown.numpy(); answered = ~t.unknown.numpy() & ~held_back(t).numpy()
    out["test_300_reference"] = {"kept_correct": int(right.sum()), "answered": int(answered.sum()),
                                 "confident_wrong": int((answered & ~right).sum())}
    return out, answers


def main():
    lists = json.load(open(W / "eye_lists.json"))
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    res, answers = {}, {}
    res["TV-12"], answers["TV-12"] = score(tv12, load("B0"), lists)
    for tag in sys.argv[2:]:
        ck = torch.load(W / f"eye_{tag}.pth", map_location="cpu", weights_only=True)
        res[tag], answers[tag] = score(ck, load(tag), lists)
        res[tag]["rules"] = dev_rules(res[tag], res["TV-12"])
        res[tag]["passes_rules"] = all(res[tag]["rules"].values())
        res[tag]["gain"] = res[tag]["dev_total"] - res["TV-12"]["dev_total"]
        res[tag]["clip"] = ck["clip_head"]["clip"]
    if "B0" in answers:   # the rebuild check: B0 must give TV-12's answers
        res["B0_same_answers_as_TV12"] = {n: int(sum((answers["B0"][n].index != answers["TV-12"][n].index)
                                                     | (answers["B0"][n].unknown != answers["TV-12"][n].unknown)))
                                          for n in answers["TV-12"]}
    cands = [t for t in sys.argv[2:] if t != "B0" and res[t]["passes_rules"] and res[t]["gain"] >= 10]
    chosen = None
    if cands:
        best = max(cands, key=lambda t: res[t]["dev_total"])
        web = [t for t in cands if t in WEB and res[best]["dev_total"] - res[t]["dev_total"] <= 5]
        chosen = max(web, key=lambda t: res[t]["dev_total"]) if web else best
    res["chosen"] = {"candidate": chosen, "rule": "8 rules pass and +10 on Valid + outside 718; highest; a web-sized one within 5 wins"}
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    for t in ["TV-12", *sys.argv[2:]]:
        r = res[t]; o = r["outside"]
        print(f"{t:6} dev {r['dev_total']} (valid {r['valid']['kept_correct']} odev {r['odev']['kept_correct']} o160 {r['o160']['kept_correct']} "
              f"asian {r['asian37']['kept_correct']}) | outside answered {o['answered']}/{o['known']} right {o['answered_right']} "
              f"conf-wrong {o['confident_wrong']} rot->fresh {o['rotten_to_fresh_answered']} | coco away {r['coco400']['turned_away']} "
              f"fruit374 {r['fruit374']['turned_away']} rotfruit {r['old_rotten_fruit_761_turned_away']} newval {r['newval']['kind_right_kept']} "
              f"old->new {r['old_dev_to_new']} weak {r['weak_total']} pair {r['pair_total']} | test {r['test_300_reference']}"
              + ("" if t == "TV-12" else f" | gain {r['gain']} rules {r['passes_rules']}"))
    if "B0_same_answers_as_TV12" in res:
        print("B0 answers different from TV-12:", res["B0_same_answers_as_TV12"])
    print("chosen:", chosen)


if __name__ == "__main__":
    main()
