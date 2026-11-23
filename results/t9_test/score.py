"""Blind test set T9 (RULES.md, addendum 3): score TV-12 and the candidate PC1 once.

    DEPP_WORK=... python results/t9_test/score.py                         # the one scoring: writes scores_once.json here
    DEPP_WORK=... python results/t9_test/score.py --rehearsal OUT.json    # development photos, to try the script only

PC1 = TV-12 with the potato re-check at threshold 0.5 (experiments/potato_check; no weight differs).
The photos stay in the working folder ($DEPP_WORK/t9/img; licences); t9_test.csv lists all 90 with their source.
The judgement uses t9_main.csv: the 89 photos left when the one that is also a T8 photo is taken out; the counts on all
90 are written next to it. Before any photo is read the scoring stops unless the SHA-256 of both lists equal the seals
in RULES.md, and when scores_once.json or the run marker scores_once.running already exists (a second run, or a run that
was interrupted: look at what happened before removing the marker by hand).
--rehearsal reads no T9 photo: it runs the same code on the 162 potato / sweet potato development photos
($DEPP_WORK/rt/img, experiments/sweetpotato_prompts/dev_features.npz ids) and marks the output as a rehearsal.
(HEAD, the earlier candidate SP1's head, is still used by results/user_photos_2/score.py.)
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


RESULT = HERE / "scores_once.json"
CANDIDATE = "PC1"
POTATO_CHECK = {"threshold": 0.5}


def conditions(tv12: dict, candidate: dict) -> dict:
    """RULES.md addendum 3 (experiments/potato_check/PREREG.md), all against TV-12 on the same photos."""
    return {"dirty fresh potato right: +6 or more": candidate["potato_fresh"]["right"] >= tv12["potato_fresh"]["right"] + 6,
            "rotten potato right: at least -2": candidate["potato_rotten"]["right"] >= tv12["potato_rotten"]["right"] - 2,
            "rotten called fresh (any vegetable): at most +2": candidate["_all"]["rotten_called_fresh"] <= tv12["_all"]["rotten_called_fresh"] + 2,
            "rotten called fresh among shown answers: not more": candidate["_all"]["rotten_called_fresh_shown"] <= tv12["_all"]["rotten_called_fresh_shown"],
            "all photos right: +3 or more": candidate["_all"]["right"] >= tv12["_all"]["right"] + 3}


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
    for name in ("t9_test.csv", "t9_main.csv"):
        sealed = re.findall(rf"{re.escape(name)} SHA-256[^`]*`([0-9a-f]{{64}})`", text)
        digest = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
        if not sealed or sealed[-1] != digest:
            raise SystemExit(f"{name} does not match its seal in RULES.md ({digest}): nothing was read or scored")
        found[name] = digest
    return found


def claim_the_one_run(result: Path, marker: Path) -> None:
    """Stop before any photo is read when T9 was scored already or a scoring run is (or was left) under way."""
    if result.exists():
        raise SystemExit(f"{result.name} exists: T9 has been scored once already; nothing was read")
    try:
        os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
    except FileExistsError:
        raise SystemExit(f"{marker.name} exists: a scoring run is under way or was interrupted; nothing was read") from None


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
    rehearsal = "--rehearsal" in sys.argv
    arguments = [a for a in sys.argv[1:] if a != "--rehearsal"]
    if rehearsal != (len(arguments) == 1):
        raise SystemExit("usage: score.py   (the one scoring, writes scores_once.json)   or   score.py --rehearsal OUT.json")
    out, marker = (Path(arguments[0]), None) if rehearsal else (RESULT, RESULT.with_suffix(".running"))
    if rehearsal and out.resolve() == RESULT.resolve():
        raise SystemExit("a rehearsal may not write scores_once.json")
    work = Path(os.environ["DEPP_WORK"])
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    if rehearsal:
        import numpy as np
        stored = np.load(REPO / "experiments/sweetpotato_prompts/dev_features.npz")
        ids, labels = [str(i) for i in stored["ids"]], [CLASS_NAMES[int(i)] for i in stored["y"]]
        severities, paths, sealed, main_ids = ["dev"] * len(ids), [work / "rt/img" / f"{i}.jpg" for i in ids], {}, set(ids)
    else:
        sealed = seals()
        claim_the_one_run(out, marker)
        rows = list(csv.DictReader(open(HERE / "t9_test.csv", encoding="utf-8")))
        ids, labels, severities = [r["id"] for r in rows], [r["class_name"] for r in rows], [r["severity"] for r in rows]
        paths = [work / "t9/img" / f"{i}.jpg" for i in ids]
        main_ids = {r["id"] for r in csv.DictReader(open(HERE / "t9_main.csv", encoding="utf-8"))}
    logits, features = model_inputs(paths)
    main = [i for i, x in enumerate(ids) if x in main_ids]
    pick = lambda values: [values[i] for i in main]
    result = {"rehearsal_on_development_photos": rehearsal, "seals": sealed, "n": len(ids), "n_main": len(main)}
    for name, checkpoint in (("TV-12", tv12), (CANDIDATE, {**tv12, "potato_check": POTATO_CHECK})):
        a = decide(logits, checkpoint, features)
        answers = [AWAY if u else CLASS_NAMES[int(i)] for i, u in zip(a.index, a.unknown)]
        held = held_back(a).tolist()
        result[name] = {"main": report(pick(labels), pick(severities), pick(answers), pick(held)),
                        "all": report(labels, severities, answers, held), "answers": dict(zip(ids, answers)),
                        "shown_probability": {i: round(float(p), 3) for i, p in zip(ids, a.probability)}}
    result["conditions"] = conditions(result["TV-12"]["main"], result[CANDIDATE]["main"])
    result["all_conditions_met"] = all(result["conditions"].values())
    result["conditions_on_all_photos"] = conditions(result["TV-12"]["all"], result[CANDIDATE]["all"])
    base, other = list(result["TV-12"]["answers"].values()), list(result[CANDIDATE]["answers"].values())
    result["right_and_wrong_against_TV-12"] = paired(pick(labels), pick(base), pick(other))
    result["differences"] = [{"id": i, "class": y, "TV-12": a, CANDIDATE: b} for i, y, a, b in zip(ids, labels, base, other) if a != b]
    partial = out.with_suffix(".partial")
    json.dump(result, open(partial, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    os.replace(partial, out)
    if marker is not None:
        marker.unlink()
    for name in ("TV-12", CANDIDATE):
        r = result[name]["main"]
        print(name, {c: f"{v['right']}/{v['n']}" for c, v in r.items() if c != "_all"}, r["_all"])
    print("conditions:", result["conditions"], "all met:", result["all_conditions_met"], "differences:", len(result["differences"]))


if __name__ == "__main__":
    main()
