"""Blind test set T5 (RULES.md): score TV-12 and the bigger_eye candidate L1 once.

The photos stay in the working folder (licences); t5_test.csv lists them with source, licence and author. The script
refuses to run unless the SHA-256 of t5_test.csv equals the seal written at the end of RULES.md.

    python results/t5_test/score.py OUT.json
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
from decision import decide, held_back  # noqa: E402
from images import make_transform, open_image  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402
from networks import input_size, load_model  # noqa: E402

IMG = S / "t5test/img"
TURNED_AWAY = "거름"
REJECT = ("rotten_fruit", "fresh_fruit", "other_veg")


def seal_ok() -> str:
    digest = hashlib.sha256((HERE / "t5_test.csv").read_bytes()).hexdigest()
    sealed = re.findall(r"SHA-256[^`]*`([0-9a-f]{64})`", (HERE / "RULES.md").read_text(encoding="utf-8"))
    assert sealed and sealed[-1] == digest, f"t5_test.csv does not match the seal in RULES.md ({digest})"
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


def score(card, ck: dict, b: np.ndarray, f: np.ndarray, labels: list[str]) -> dict:
    a = decide(torch.tensor(b), ck, torch.tensor(f))
    answers = [TURNED_AWAY if u else CLASS_NAMES[int(i)] for i, u in zip(a.index, a.unknown)]
    held = held_back(a).numpy()
    five = [i for i, y in enumerate(labels) if y in CLASS_NAMES[:10]]
    c = card(labels, answers)
    right = np.array([answers[i] == labels[i] for i in five]); answered = ~a.unknown.numpy()[five] & ~held[five]
    rotten = np.array([labels[i].endswith("_rotten") for i in five])
    answered_rotten_to_fresh = sum(answered[k] and rotten[k] and answers[i].endswith("_fresh") for k, i in enumerate(five))
    newv = [i for i, y in enumerate(labels) if y in CLASS_NAMES[10:]]
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


def main():
    digest = seal_ok()
    card = _load("make_scorecard", REPO / "results/scorecard/make_scorecard.py").card
    rows = list(csv.DictReader(open(HERE / "t5_test.csv", encoding="utf-8")))
    labels = [r["class_name"] for r in rows]
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    l1 = torch.load(W / "eye_L1.pth", map_location="cpu", weights_only=True)
    clips = {"TV-12": ClipModel("cpu"), "L1": ClipModel("cpu", **l1["clip_head"]["clip"])}
    b, F = outputs([str(IMG / f"{r['id']}.jpg") for r in rows], clips)
    res = {"seal": digest, "n": len(rows), "L1_image_model": l1["clip_head"]["clip"],
           "TV-12": score(card, tv12, b, F["TV-12"], labels), "L1": score(card, l1, b, F["L1"], labels)}
    t, c = res["TV-12"], res["L1"]
    d = lambda part, key: c[part][key] - t[part][key]
    res["L1_rules"] = {"five-vegetable kept >= TV-12": d("five_vegetables", "right_and_kept") >= 0,
                       "five-vegetable rotten -> fresh <= TV-12": d("five_vegetables", "rotten_to_fresh") <= 0,
                       "not-five turned away >= -2": d("not_five", "turned_away") >= -2,
                       "new-vegetable kind kept >= -1": d("new_vegetables", "kind_right_and_kept") >= -1,
                       "five vegetables to new vegetables <= +1": d("five_vegetables", "to_new_vegetable") <= 1}
    res["L1_proposed"] = all(res["L1_rules"].values())
    res["differences"] = [{"id": r["id"], "class": r["class_name"], "TV-12": x, "L1": y}
                          for r, x, y in zip(rows, t["answers"], c["answers"]) if x != y]
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    for m in ("TV-12", "L1"):
        r = res[m]; f = r["five_vegetables"]
        print(f"{m:6} five {f['right_and_kept']}/{f['n']} {f['ci95']} rot->fresh {f['rotten_to_fresh']}/{f['rotten_n']} "
              f"fresh->rot {f['fresh_to_rotten']}/{f['fresh_n']} kind {f['wrong_kind']} away {f['turned_away']} hold {f['hold_back']} | "
              f"new {r['new_vegetables']['kind_right_and_kept']}/{r['new_vegetables']['n']} | not-five {r['not_five']}")
    print(json.dumps(res["L1_rules"]), "proposed:", res["L1_proposed"], "differences:", len(res["differences"]))


if __name__ == "__main__":
    main()
