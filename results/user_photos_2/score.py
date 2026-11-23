"""User photos, batch 2 (RULES.md): a reference check of TV-12 and SP1, not an adoption test. Scored once.

    DEPP_WORK=... python results/user_photos_2/score.py OUT.json

Reads photos.csv; only rows with status "scored" / "scored_flagged" are given to the models ($DEPP_WORK/user2/img).
Photos that are copies of T9 or SP1 training photos were never copied there. Counting as in results/t9_test/score.py.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "results/t9_test"))
from decision import decide, held_back  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402
from score import AWAY, HEAD, model_inputs, paired, report  # noqa: E402


def main():
    rows = [r for r in csv.DictReader(open(HERE / "photos.csv", encoding="utf-8")) if r["status"].startswith("scored")]
    labels, severities = [r["class_name"] for r in rows], ["flagged" if r["status"] == "scored_flagged" else "clean" for r in rows]
    logits, features = model_inputs([Path(os.environ["DEPP_WORK"]) / "user2/img" / f"{r['id']}.jpg" for r in rows])
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    sp1 = {**tv12, "new_vegetable_head": torch.load(HEAD, map_location="cpu", weights_only=True)}
    clean = [i for i, s in enumerate(severities) if s == "clean"]
    result = {"reference_check_not_an_adoption_test": True, "n": len(rows), "n_clean": len(clean)}
    for name, checkpoint in (("TV-12", tv12), ("SP1", sp1)):
        a = decide(logits, checkpoint, features)
        answers = [AWAY if u else CLASS_NAMES[int(i)] for i, u in zip(a.index, a.unknown)]
        held = held_back(a).tolist()
        pick = lambda values: [values[i] for i in clean]
        result[name] = {"all": report(labels, severities, answers, held), "clean": report(pick(labels), pick(severities), pick(answers), pick(held)),
                        "answers": {r["id"]: x for r, x in zip(rows, answers)}}
    base, other = list(result["TV-12"]["answers"].values()), list(result["SP1"]["answers"].values())
    result["right_and_wrong_against_TV-12"] = paired(labels, base, other)
    result["differences"] = [{"id": r["id"], "class": r["class_name"], "TV-12": x, "SP1": y} for r, x, y in zip(rows, base, other) if x != y]
    json.dump(result, open(sys.argv[1], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for name in ("TV-12", "SP1"):
        for part in ("clean", "all"):
            print(name, part, {c: v["answers"] for c, v in result[name][part].items() if c != "_all"}, {k: result[name][part]["_all"][k] for k in ("n", "right", "held_back", "shown", "shown_right")})
    print("differences:", result["differences"])


if __name__ == "__main__":
    main()
