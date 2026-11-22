"""Blind test set T9 (RULES.md): score TV-12 and the candidate SP1 once.

    DEPP_WORK=... python results/t9_test/score.py OUT.json
    DEPP_WORK=... python results/t9_test/score.py OUT.json --rehearsal    # development photos, to try the script only

SP1 = model.pth with the new-vegetable head of experiments/sweetpotato_photos/heads_SP1.pt (nothing else differs).
The photos stay in the working folder ($DEPP_WORK/t9/img; licences); t9_test.csv lists them with their source.
Refuses to run unless the SHA-256 of t9_test.csv and of heads_SP1.pt equal the seals in RULES.md.
--rehearsal reads no T9 photo: it runs the same code on the 162 potato / sweet potato development photos
($DEPP_WORK/rt/img, experiments/sweetpotato_prompts/dev_features.npz ids) and marks the output as a rehearsal.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from decision import decide, held_back  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402

AWAY = "turned_away"
HEAD = REPO / "experiments/sweetpotato_photos/heads_SP1.pt"
POTATO = ("potato_rotten", "potato_fresh")


def kind(name: str) -> str:
    return name.rsplit("_", 1)[0]


def says_fresh(answer: str) -> bool:
    """The answer claims a fresh vegetable (a green onion answer claims no condition)."""
    return answer.endswith("_fresh") and not answer.startswith("greenonion")


def right(label: str, answer: str) -> bool:
    """RULES.md: a sweet potato photo is right when the answer is sweet potato (either condition); a potato photo
    when the answer is its class."""
    return kind(answer) == "sweetpotato" if label.startswith("sweetpotato") else answer == label


def report(labels: list[str], severities: list[str], answers: list[str], held: list[bool]) -> dict:
    out = {}
    for c in sorted(set(labels)):
        idx = [i for i, y in enumerate(labels) if y == c]
        entry = {"n": len(idx), "right": sum(right(c, answers[i]) for i in idx), "class_right": sum(answers[i] == c for i in idx),
                 "answered_sweet_potato": sum(kind(answers[i]) == "sweetpotato" for i in idx),
                 "answered_potato": sum(kind(answers[i]) == "potato" for i in idx),
                 "wrong_condition_same_kind": sum(kind(answers[i]) == kind(c) and answers[i] != c for i in idx),
                 "turned_away": sum(answers[i] == AWAY for i in idx), "held_back": sum(held[i] for i in idx),
                 "answers": {a: sum(answers[i] == a for i in idx) for a in sorted({answers[i] for i in idx})}}
        if c.endswith("_rotten"):
            entry["called_fresh"] = sum(says_fresh(answers[i]) for i in idx)
        for s in sorted({severities[i] for i in idx}):
            part = [i for i in idx if severities[i] == s]
            entry[f"right_{s}"] = [sum(right(c, answers[i]) for i in part), len(part)]
        out[c] = entry
    n = len(labels)
    rotten = [i for i in range(n) if labels[i].endswith("_rotten")]
    shown = [i for i in range(n) if answers[i] != AWAY and not held[i]]
    out["_all"] = {"n": n, "right": sum(right(labels[i], answers[i]) for i in range(n)),
                   "rotten_n": len(rotten), "rotten_called_fresh": sum(says_fresh(answers[i]) for i in rotten),
                   "rotten_called_fresh_shown": sum(says_fresh(answers[i]) for i in rotten if not held[i]),
                   "potato_n": sum(y in POTATO for y in labels),
                   "potato_answered_sweet_potato": sum(kind(answers[i]) == "sweetpotato" for i in range(n) if labels[i] in POTATO),
                   "turned_away": answers.count(AWAY), "held_back": sum(held),
                   "shown": len(shown), "shown_right": sum(right(labels[i], answers[i]) for i in shown)}
    return out


def conditions(tv12: dict, sp1: dict) -> dict:
    """RULES.md (experiments/sweetpotato_photos/PREREG.md, addendum 2), all against TV-12 on the same photos."""
    return {"rotten sweet potato answered sweet potato: +6 or more": sp1["sweetpotato_rotten"]["right"] >= tv12["sweetpotato_rotten"]["right"] + 6,
            "potatoes answered sweet potato: at most +2": sp1["_all"]["potato_answered_sweet_potato"] <= tv12["_all"]["potato_answered_sweet_potato"] + 2,
            "rotten potato right: at least -1": sp1["potato_rotten"]["right"] >= tv12["potato_rotten"]["right"] - 1,
            "rotten called fresh (any vegetable): not more": sp1["_all"]["rotten_called_fresh"] <= tv12["_all"]["rotten_called_fresh"],
            "all photos right: not fewer": sp1["_all"]["right"] >= tv12["_all"]["right"]}


def paired(labels: list[str], base: list[str], other: list[str]) -> dict:
    out = {}
    for y, a, b in zip(labels, base, other):
        counts = out.setdefault(y, {"right_to_wrong": 0, "wrong_to_right": 0})
        counts["right_to_wrong"] += right(y, a) and not right(y, b)
        counts["wrong_to_right"] += not right(y, a) and right(y, b)
    return out


def seals() -> dict:
    text = (HERE / "RULES.md").read_text(encoding="utf-8")
    found = {}
    for name, path in (("t9_test.csv", HERE / "t9_test.csv"), ("heads_SP1.pt", HEAD)):
        sealed = re.findall(rf"{re.escape(name)} SHA-256[^`]*`([0-9a-f]{{64}})`", text)
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        assert sealed and sealed[-1] == digest, f"{name} does not match its seal in RULES.md ({digest})"
        found[name] = digest
    return found


def model_inputs(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor]:
    from clip_model import ClipModel
    from images import make_transform, open_image
    from networks import input_size, load_model
    model, checkpoint = load_model(REPO / "model.pth", torch.device("cpu"))
    transform, clip = make_transform(input_size(checkpoint)), ClipModel("cpu")
    logits, features = [], []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            images = [open_image(p) for p in paths[i:i + 32]]
            logits.append(model(torch.stack([transform(image) for image in images])))
            features.append(clip.image_features(images).cpu())
    return torch.cat(logits), torch.cat(features)


def main():
    out, rehearsal = Path(sys.argv[1]), "--rehearsal" in sys.argv
    work = Path(os.environ["DEPP_WORK"])
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    if rehearsal:
        import numpy as np
        stored = np.load(REPO / "experiments/sweetpotato_prompts/dev_features.npz")
        ids, labels = [str(i) for i in stored["ids"]], [CLASS_NAMES[int(i)] for i in stored["y"]]
        severities, paths, sealed = ["dev"] * len(ids), [work / "rt/img" / f"{i}.jpg" for i in ids], {}
        head = torch.load(HEAD, map_location="cpu", weights_only=True) if HEAD.exists() else tv12["new_vegetable_head"]
    else:
        sealed = seals()
        rows = list(csv.DictReader(open(HERE / "t9_test.csv", encoding="utf-8")))
        ids, labels, severities = [r["id"] for r in rows], [r["class_name"] for r in rows], [r["severity"] for r in rows]
        paths = [work / "t9/img" / f"{i}.jpg" for i in ids]
        head = torch.load(HEAD, map_location="cpu", weights_only=True)
    logits, features = model_inputs(paths)
    result = {"rehearsal_on_development_photos": rehearsal, "seals": sealed, "n": len(ids)}
    for name, checkpoint in (("TV-12", tv12), ("SP1", {**tv12, "new_vegetable_head": head})):
        a = decide(logits, checkpoint, features)
        answers = [AWAY if u else CLASS_NAMES[int(i)] for i, u in zip(a.index, a.unknown)]
        result[name] = {**report(labels, severities, answers, held_back(a).tolist()), "answers": dict(zip(ids, answers))}
    result["conditions"] = conditions(result["TV-12"], result["SP1"])
    result["all_conditions_met"] = all(result["conditions"].values())
    base, other = list(result["TV-12"]["answers"].values()), list(result["SP1"]["answers"].values())
    result["right_and_wrong_against_TV-12"] = paired(labels, base, other)
    result["differences"] = [{"id": i, "class": y, "TV-12": a, "SP1": b} for i, y, a, b in zip(ids, labels, base, other) if a != b]
    json.dump(result, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    for name in ("TV-12", "SP1"):
        print(name, {c: f"{v['right']}/{v['n']}" for c, v in result[name].items() if c not in ("_all", "answers")}, result[name]["_all"])
    print("conditions:", result["conditions"], "all met:", result["all_conditions_met"], "differences:", len(result["differences"]))


if __name__ == "__main__":
    main()
