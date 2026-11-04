"""Blind test set T6 (RULES.md): score TV-12 and one fix candidate once, after the candidate is chosen on development photos
and its T6 rule is committed.

The photos stay in the working folder (licences); t6_test.csv lists them with source, licence and author. The script
refuses to run unless the SHA-256 of t6_test.csv equals the seal written at the end of RULES.md.

    python results/t6_test/score.py CANDIDATE.pth OUT.json
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments/watermark"))
from common import S, W  # noqa: E402
from clip_model import ClipModel  # noqa: E402
from decision import apply_crop_check, decide, held_back, needs_crop_check  # noqa: E402
from images import five_crops, make_transform, open_image  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402
from networks import input_size, load_model  # noqa: E402

IMG = S / "t6test/img"
TURNED_AWAY = "거름"
REJECT = ("rotten_fruit", "fresh_fruit", "other_veg")


def seal_ok() -> str:
    digest = hashlib.sha256((HERE / "t6_test.csv").read_bytes()).hexdigest()
    sealed = re.findall(r"SHA-256[^`]*`([0-9a-f]{64})`", (HERE / "RULES.md").read_text(encoding="utf-8"))
    assert sealed and sealed[-1] == digest, f"t6_test.csv does not match the seal in RULES.md ({digest})"
    return digest


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outputs(paths: list[str], clips: dict) -> tuple[np.ndarray, dict]:
    """TV-12's vegetable-model logits (the same in both systems) and the image features of each image model."""
    model, ck = load_model(REPO / "model.pth", torch.device("cpu")); tf = make_transform(input_size(ck))
    B, F = [], {k: [] for k in clips}
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            ims = [open_image(p) for p in paths[i:i + 32]]
            B.append(model(torch.stack([tf(im) for im in ims])))
            for k, clip in clips.items():
                F[k].append(clip.image_features(ims).cpu())
    return torch.cat(B).numpy(), {k: torch.cat(v).numpy() for k, v in F.items()}


def crop_check(a, ck: dict, paths: list[str], clip):
    """RULES.md addendum (2026-11-04): the tomato crop check of experiments/tomato_crops, when the checkpoint has it."""
    rows = torch.where(needs_crop_check(a, ck))[0]
    if len(rows) == 0:
        return a
    with torch.inference_mode():
        fc = torch.stack([clip.image_features(five_crops(open_image(paths[i]))).cpu() for i in rows.tolist()])
    return apply_crop_check(a, ck, fc, rows)


def score(card, ck: dict, b: np.ndarray, f: np.ndarray, labels: list[str], paths: list[str] | None = None, clip=None) -> dict:
    a = decide(torch.tensor(b), ck, torch.tensor(f))
    if "tomato_crop_check" in ck:
        a = crop_check(a, ck, paths, clip)
    answers = [TURNED_AWAY if u else CLASS_NAMES[int(i)] for i, u in zip(a.index, a.unknown)]
    held = held_back(a).numpy()
    five = [i for i, y in enumerate(labels) if y in CLASS_NAMES[:10]]
    c = card(labels, answers)
    right = np.array([answers[i] == labels[i] for i in five]); answered = ~a.unknown.numpy()[five] & ~held[five]
    rotten = np.array([labels[i].endswith("_rotten") for i in five])
    answered_rotten_to_fresh = sum(answered[k] and rotten[k] and answers[i].endswith("_fresh") for k, i in enumerate(five))
    untrained = set(ck.get("new_vegetable_head", {}).get("untrained_classes", []))
    newv = [i for i, y in enumerate(labels) if y in CLASS_NAMES[10:] and not ({"greenonion_fresh", "greenonion_rotten"} <= untrained and y.startswith("greenonion"))]
    rej = [i for i, y in enumerate(labels) if y in REJECT]
    kind = lambda s: s.rsplit("_", 1)[0]
    return {
        "five_vegetables": {k: c[k] for k in ("n", "right_and_kept", "accuracy", "ci95", "turned_away", "wrong_kind", "rotten_n",
                                              "rotten_to_fresh", "fresh_n", "fresh_to_rotten", "per_class")}
                           | {"to_new_vegetable": sum(answers[i] != TURNED_AWAY and CLASS_NAMES.index(answers[i]) >= 10 for i in five),
                              "hold_back": {"answered": int(answered.sum()), "answered_right": int((answered & right).sum()),
                                            "confident_wrong": int((answered & ~right).sum()),
                                            "rotten_to_fresh_answered": int(answered_rotten_to_fresh)}},
        "new_vegetables": {"n": len(newv), "kind_right_and_kept": sum(answers[i] != TURNED_AWAY and kind(answers[i]) == kind(labels[i]) for i in newv)},
        "not_five": {"n": len(rej), "turned_away": sum(answers[i] == TURNED_AWAY for i in rej),
                     **{k: f"{sum(answers[i] == TURNED_AWAY for i in rej if labels[i] == k)}/{sum(labels[i] == k for i in rej)}" for k in REJECT}},
        "answers": answers,
    }


WEAK = {"fresh potato -> rotten potato": ("potato_fresh", "potato_rotten"),
        "rotten fruit -> rotten potato (should be turned away)": ("rotten_fruit", "potato_rotten"),
        "rotten tomato -> fresh tomato": ("tomato_rotten", "tomato_fresh")}


def weak_spots(rows: list[dict], answers: list[str]) -> dict:
    """The three repeated weak spots (RULES.md), with the rotten tomatoes split by source (internet / dataset)."""
    out = {}
    for name, (label, wrong) in WEAK.items():
        idx = [i for i, r in enumerate(rows) if r["class_name"] == label]
        good = sum(answers[i] == (TURNED_AWAY if label in REJECT else label) for i in idx)
        out[name] = {"n": len(idx), "good": good, "this_error": sum(answers[i] == wrong for i in idx)}
    for src in ("dataset", "internet"):
        idx = [i for i, r in enumerate(rows) if r["class_name"] == "tomato_rotten" and (r["source"] == "dataset") == (src == "dataset")]
        out[f"rotten tomato ({src})"] = {"n": len(idx), "good": sum(answers[i] == "tomato_rotten" for i in idx),
                                         "this_error": sum(answers[i] == "tomato_fresh" for i in idx)}
    return out


def main():
    digest = seal_ok()
    card = _load("make_scorecard", REPO / "results/scorecard/make_scorecard.py").card
    rows = list(csv.DictReader(open(HERE / "t6_test.csv", encoding="utf-8")))
    missing = [r["id"] for r in rows if not (IMG / f"{r['id']}.jpg").exists()]   # RULES.md addendum: left out for both models
    rows = [r for r in rows if r["id"] not in missing]
    labels = [r["class_name"] for r in rows]
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    cand = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    clips = {"TV-12": ClipModel("cpu")}
    if cand["clip_head"]["clip"] != tv12["clip_head"]["clip"]:
        clips["candidate"] = ClipModel("cpu", **cand["clip_head"]["clip"])
    paths = [str(IMG / f"{r['id']}.jpg") for r in rows]
    b, F = outputs(paths, clips)
    F.setdefault("candidate", F["TV-12"])
    res = {"seal": digest, "n": len(rows), "missing_left_out": missing, "candidate": str(sys.argv[1]), "candidate_version": cand.get("version"),
           "candidate_tomato_crop_check": cand.get("tomato_crop_check"),
           "TV-12": score(card, tv12, b, F["TV-12"], labels),
           "candidate_scores": score(card, cand, b, F["candidate"], labels, paths, clips.get("candidate", clips["TV-12"]))}
    for m in ("TV-12", "candidate_scores"):
        res[m]["weak_spots"] = weak_spots(rows, res[m]["answers"])
    res["differences"] = [{"id": r["id"], "class": r["class_name"], "TV-12": x, "candidate": y}
                          for r, x, y in zip(rows, res["TV-12"]["answers"], res["candidate_scores"]["answers"]) if x != y]
    t, c = res["TV-12"], res["candidate_scores"]
    right = lambda r, cls: sum(a == cls for a, y in zip(r["answers"], labels) if y == cls)
    res["green_onion_rows"] = [{"id": r["id"], "TV-12": t["answers"][i], "candidate": c["answers"][i]} for i, r in enumerate(rows) if r["class_name"].startswith("greenonion")]
    res["conditions_tomato_crops"] = {   # experiments/tomato_crops/PREREG.md addendum (2026-11-04)
        "rotten tomato +3": right(c, "tomato_rotten") - right(t, "tomato_rotten") >= 3,
        "fresh tomato >= -1": right(c, "tomato_fresh") - right(t, "tomato_fresh") >= -1,
        "five vegetables right and kept >= TV-12": c["five_vegetables"]["right_and_kept"] >= t["five_vegetables"]["right_and_kept"],
        "turned away of 75 >= -3": c["not_five"]["turned_away"] - t["not_five"]["turned_away"] >= -3,
        "onion kind >= -1": c["new_vegetables"]["kind_right_and_kept"] - t["new_vegetables"]["kind_right_and_kept"] >= -1}
    res["tomato_counts"] = {m: {cls: f"{right(r, cls)}/{labels.count(cls)}" for cls in ("tomato_fresh", "tomato_rotten")} for m, r in (("TV-12", t), ("candidate", c))}
    res["proposed"] = all(res["conditions_tomato_crops"].values())
    json.dump(res, open(sys.argv[2], "w"), indent=1, ensure_ascii=False)
    for m in ("TV-12", "candidate_scores"):
        r = res[m]; f = r["five_vegetables"]
        print(f"{m:16} five {f['right_and_kept']}/{f['n']} rot->fresh {f['rotten_to_fresh']}/{f['rotten_n']} fresh->rot {f['fresh_to_rotten']}/{f['fresh_n']} "
              f"| not-five away {r['not_five']['turned_away']}/{r['not_five']['n']} | weak {r['weak_spots']}")
    print("differences:", len(res["differences"]), "| missing left out:", res["missing_left_out"])
    print("tomatoes:", res["tomato_counts"], "| conditions:", res["conditions_tomato_crops"], "| proposed:", res["proposed"])


if __name__ == "__main__":
    main()
