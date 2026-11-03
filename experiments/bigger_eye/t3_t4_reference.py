"""experiments/bigger_eye (PREREG.md, step 4 as corrected): the chosen candidate once on T3 and T4, both reference only
(T4 was scored once on 2026-11-02, T3's per-class results were used for diagnosis).

    python experiments/bigger_eye/t3_t4_reference.py L1 experiments/bigger_eye/t3_t4_reference.json

Both scripts refuse to run unless the answer lists match their seals. TV-12 is recomputed on the same photos and must give the
answers stored when T3 and T4 were scored (pipeline check). Then the candidate's answers are scored the same way.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments/watermark"))
from common import W  # noqa: E402
from clip_model import ClipModel  # noqa: E402
from decision import decide  # noqa: E402
from images import open_image  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402
from networks import input_size, load_model  # noqa: E402
from images import make_transform  # noqa: E402

TURNED_AWAY = "거름"
REJECT = ("rotten_fruit", "fresh_fruit", "other_veg")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def outputs(paths: list[str], clips: dict) -> tuple[np.ndarray, dict]:
    """TV-12's vegetable-model logits, and image features from each image model."""
    model, ck = load_model(REPO / "model.pth", torch.device("cpu")); tf = make_transform(input_size(ck))
    B, F = [], {k: [] for k in clips}
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            ims = [open_image(p) for p in paths[i:i + 32]]
            B.append(model(torch.stack([tf(im) for im in ims])))
            for k, clip in clips.items():
                F[k].append(clip.image_features(ims).cpu())
    return torch.cat(B).numpy(), {k: torch.cat(v).numpy() for k, v in F.items()}


def names(ck: dict, b: np.ndarray, f: np.ndarray) -> list[str]:
    a = decide(torch.tensor(b), ck, torch.tensor(f))
    return [TURNED_AWAY if u else CLASS_NAMES[int(i)] for i, u in zip(a.index, a.unknown)]


def summary(card, labels: list[str], answers: list[str]) -> dict:
    c = card(labels, answers)
    newv = [(y, a) for y, a in zip(labels, answers) if y in CLASS_NAMES[10:]]
    rej = [(y, a) for y, a in zip(labels, answers) if y in REJECT]
    return {"five_vegetables": {k: c[k] for k in ("n", "right_and_kept", "accuracy", "ci95", "turned_away", "wrong_kind",
                                                  "rotten_n", "rotten_to_fresh", "fresh_n", "fresh_to_rotten", "per_class")},
            "new_vegetables": {"n": len(newv), "kind_right_and_kept": sum(a != TURNED_AWAY and a.rsplit("_", 1)[0] == y.rsplit("_", 1)[0] for y, a in newv)},
            "not_five": {"n": len(rej), "turned_away": sum(a == TURNED_AWAY for _, a in rej)},
            "answers": answers}


def main():
    tag, out = sys.argv[1], Path(sys.argv[2])
    card = _load("make_scorecard", REPO / "results/scorecard/make_scorecard.py").card
    t3 = _load("t3_score", REPO / "results/t3_test/score.py"); t4 = _load("t4_score", REPO / "results/t4_test/score.py")
    seals = {"t3": t3.seal_ok(), "t4": t4.seal_ok()}
    rows = {"t3": list(csv.DictReader(open(REPO / "results/t3_test/t3_test.csv", encoding="utf-8"))),
            "t4": list(csv.DictReader(open(REPO / "results/t4_test/t4_test.csv", encoding="utf-8")))}
    paths = {"t3": [str(t3.IMG / f"{r['id']}.jpg") for r in rows["t3"]], "t4": [str(t4.IMG / f"{r['id']}.jpg") for r in rows["t4"]]}
    stored = {"t3": json.load(open(REPO / "results/t3_test/scores_once.json"))["ZS8"]["answers"],
              "t4": json.load(open(REPO / "results/t4_test/scores_reference.json"))["TV-12"]["answers"]}
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    cand = torch.load(W / f"eye_{tag}.pth", map_location="cpu", weights_only=True)
    clips = {"TV-12": ClipModel("cpu"), tag: ClipModel("cpu", **cand["clip_head"]["clip"])}
    res = {"candidate": tag, "image_model": cand["clip_head"]["clip"], "seals": seals,
           "note": "reference only: T4 was scored once before (2026-11-02) and T3's per-class results were used for diagnosis"}
    for s in ("t3", "t4"):
        b, F = outputs(paths[s], clips)
        again = names(tv12, b, F["TV-12"])
        res[f"{s}_pipeline_check_TV12_same_as_stored"] = sum(x != y for x, y in zip(again, stored[s]))
        labels = [r["class_name"] for r in rows[s]]
        res[s] = {"TV-12": summary(card, labels, stored[s]), tag: summary(card, labels, names(cand, b, F[tag]))}
    k = lambda m, key: sum(res[s][m]["five_vegetables"][key] for s in ("t3", "t4"))
    res["rule"] = {"kept_right_t3_t4": {m: k(m, "right_and_kept") for m in ("TV-12", tag)},
                   "rotten_to_fresh_t3_t4": {m: k(m, "rotten_to_fresh") for m in ("TV-12", tag)}}
    res["rule"]["passes"] = (res["rule"]["kept_right_t3_t4"][tag] >= res["rule"]["kept_right_t3_t4"]["TV-12"]
                             and res["rule"]["rotten_to_fresh_t3_t4"][tag] <= res["rule"]["rotten_to_fresh_t3_t4"]["TV-12"])
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)
    for s in ("t3", "t4"):
        print(s, "pipeline differences:", res[f"{s}_pipeline_check_TV12_same_as_stored"])
        for m in ("TV-12", tag):
            r = res[s][m]; f = r["five_vegetables"]
            print(f"  {m:6} five {f['right_and_kept']}/{f['n']} rot->fresh {f['rotten_to_fresh']}/{f['rotten_n']} fresh->rot "
                  f"{f['fresh_to_rotten']}/{f['fresh_n']} kind {f['wrong_kind']} away {f['turned_away']} | new veg "
                  f"{r['new_vegetables']['kind_right_and_kept']}/{r['new_vegetables']['n']} | not-five away {r['not_five']['turned_away']}/{r['not_five']['n']}")
    print(json.dumps(res["rule"], ensure_ascii=False))


if __name__ == "__main__":
    main()
