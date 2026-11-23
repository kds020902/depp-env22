"""experiments/potato_check (PREREG.md): TV-12 against PC1 (TV-12 + the potato re-check, threshold 0.5) on development photos.

    python experiments/potato_check/evaluate.py                      # from the stored inputs
    DEPP_WORK=... python experiments/potato_check/evaluate.py --store   # first make user2_dev_features.npz from the photos

1) the development sets of tools/compare_on_dev.py (already looked at when the threshold was chosen: a safety check);
2) the 41 scored photos of results/user_photos_2 (PC1 has not seen them): stored as model inputs in user2_dev_features.npz.
No blind test set is read. Writes dev_results.json.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for folder in ("src", "tools", "results/t9_test"):
    sys.path.insert(0, str(REPO / folder))
from compare_on_dev import changed, development_sets, evaluate, paired  # noqa: E402
from decision import decide, held_back  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402
from score import AWAY, model_inputs, report, says_fresh  # noqa: E402

THRESHOLD = 0.5
USER2 = HERE / "user2_dev_features.npz"
FIVE = CLASS_NAMES[:10]


def store_user2():
    rows = [r for r in csv.DictReader(open(REPO / "results/user_photos_2/photos.csv", encoding="utf-8")) if r["status"].startswith("scored")]
    logits, features = model_inputs([Path(os.environ["DEPP_WORK"]) / "user2/img" / f"{r['id']}.jpg" for r in rows])
    np.savez_compressed(USER2, b=logits.numpy().astype(np.float32), X=features.numpy().astype(np.float32),
                        y=np.array([CLASS_NAMES.index(r["class_name"]) for r in rows], np.int16), ids=np.array([r["id"] for r in rows]),
                        flagged=np.array([r["status"] == "scored_flagged" for r in rows]))


def user2_report(checkpoint: dict) -> dict:
    stored = np.load(USER2)
    labels = [CLASS_NAMES[int(i)] for i in stored["y"]]
    a = decide(torch.tensor(stored["b"]), checkpoint, torch.tensor(stored["X"]))
    answers = [AWAY if u else CLASS_NAMES[int(i)] for i, u in zip(a.index, a.unknown)]
    held = held_back(a).tolist()
    result = report(labels, ["flagged" if f else "clean" for f in stored["flagged"]], answers, held)
    rotten_sweet = [i for i, y in enumerate(labels) if y == "sweetpotato_rotten"]
    result["_rotten_sweet_potato_called_fresh"] = sum(says_fresh(answers[i]) for i in rotten_sweet)
    result["answers"] = dict(zip([str(i) for i in stored["ids"]], answers))
    result["shown_probability"] = {str(i): round(float(p), 3) for i, p in zip(stored["ids"], a.probability)}
    return result


def main():
    if "--store" in sys.argv:
        store_user2()
    # model.pth is TV-13 = PC1 since the adoption; TV-12 is the same file without the potato check
    tv12 = {k: v for k, v in torch.load(REPO / "model.pth", map_location="cpu", weights_only=True).items() if k != "potato_check"}
    pc1 = {**tv12, "potato_check": {"threshold": THRESHOLD}}
    sets = development_sets()
    dev = {"TV-12": evaluate(tv12, sets), "PC1": evaluate(pc1, sets)}
    user2 = {"TV-12": user2_report(tv12), "PC1": user2_report(pc1)}
    five = lambda r: sum(r["_all_labelled"]["classes"][c]["right"] for c in FIVE)
    diff = changed(dev["TV-12"], dev["PC1"], sets)
    unlabelled = sum(n for name, n in diff.items() if sets[name][2] is None)
    base, cand = dev["TV-12"]["_all_labelled"], dev["PC1"]["_all_labelled"]
    conditions = {
        "1) rotten called fresh (any vegetable): at most +1": cand["rotten_called_fresh"] <= base["rotten_called_fresh"] + 1,
        "1) five vegetables right: not fewer": five(dev["PC1"]) >= five(dev["TV-12"]),
        "1) answers changed on unlabelled photos: at most 3": unlabelled <= 3,
        "2) dirty fresh potato right: +5 or more": user2["PC1"]["potato_fresh"]["right"] >= user2["TV-12"]["potato_fresh"]["right"] + 5,
        "2) rotten sweet potato called fresh: at most +1": user2["PC1"]["_rotten_sweet_potato_called_fresh"] <= user2["TV-12"]["_rotten_sweet_potato_called_fresh"] + 1}
    record = {"threshold": THRESHOLD, "conditions": conditions, "go_to_T9": all(conditions.values()),
              "development_sets": {"systems": dev, "answers_changed": diff, "right_and_wrong_against_TV-12": paired(dev["TV-12"], dev["PC1"], sets),
                                   "five_vegetables_right": {m: five(r) for m, r in dev.items()}},
              "user_photos_2": user2}
    json.dump(record, open(HERE / "dev_results.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for m in ("TV-12", "PC1"):
        t = dev[m]["_all_labelled"]
        print(m, "dev: right", t["right"], "five", five(dev[m]), "rotten->fresh", t["rotten_called_fresh"], "shown", t["rotten_called_fresh_shown"],
              "held", t["held_back"], "| potato fresh", dev[m]["_all_labelled"]["classes"]["potato_fresh"]["right"], "rotten", dev[m]["_all_labelled"]["classes"]["potato_rotten"]["right"])
        u = user2[m]
        print("   user2: dirty potato", u["potato_fresh"]["answers"], "held", u["potato_fresh"]["held_back"], "| rotten sweet potato", u["sweetpotato_rotten"]["answers"])
    print("answers changed:", diff)
    print(conditions, "go to T9:", record["go_to_T9"])


if __name__ == "__main__":
    main()
