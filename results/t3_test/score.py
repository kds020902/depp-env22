"""Sealed test T3 (RULES.md): score TV-11 and the ZS8 candidate once.

The photos stay in the working folder (licences); t3_test.csv lists them with source, licence and author. The script
refuses to run unless the SHA-256 of t3_test.csv equals the seal written at the end of RULES.md.

    python results/t3_test/score.py   -> scores_once.json
"""

from __future__ import annotations

import collections
import csv
import hashlib
import importlib.util
import json
import math
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
from model_utils import CLASS_NAMES, input_size, load_model, make_transform, open_image  # noqa: E402

IMG = S / "t3/img"
REJECT = ("rotten_fruit", "fresh_fruit", "other_veg")
NAMES_KO = {"cucumber": "오이", "potato": "감자", "tomato": "토마토", "bellpepper": "파프리카", "carrot": "당근",
            "onion": "양파", "sweetpotato": "고구마", "greenonion": "대파"}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


zs = _load("zs_evaluate", REPO / "experiments/zero_shot_reject/evaluate.py")


def seal_ok() -> str:
    digest = hashlib.sha256((HERE / "t3_test.csv").read_bytes()).hexdigest()
    sealed = re.findall(r"SHA-256[^`]*`([0-9a-f]{64})`", (HERE / "RULES.md").read_text(encoding="utf-8"))
    assert sealed and sealed[-1] == digest, f"t3_test.csv does not match the seal in RULES.md ({digest})"
    return digest


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def features(paths: list[str]):
    """CLIP features (same model as the CLIP heads) and vegetable-model logits of TV-11 (model.pth)."""
    from clip_model import ClipModel
    clip = ClipModel("cpu")
    model, checkpoint = load_model(REPO / "model.pth", torch.device("cpu")); tf = make_transform(input_size(checkpoint))
    F, B = [], []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            ims = [open_image(p) for p in paths[i:i + 32]]
            F.append(clip.image_features(ims).cpu()); B.append(model(torch.stack([tf(im) for im in ims])))
    return torch.cat(F).numpy(), torch.cat(B).numpy()


def pipeline_check():
    """The same code must give the cached features of the watermark round's final scoring for 32 Test photos."""
    test = [r for r in csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")) if r["split"] == "test"][:32]
    f, b = features([str(REPO / "dataset" / r["image_path"]) for r in test])
    df = float(np.abs(f - np.load(W / "final_clip.npz")["test_300"][:32]).max())
    db = float(np.abs(b - np.load(W / "final_b_WC.npz")["test_300"][:32]).max())
    assert df < 1e-3 and db < 1e-3, (df, db)
    return {"clip_max_diff": df, "vegetable_model_max_diff": db}


def answer_name(i: int, unknown: bool) -> str:
    return "거름" if unknown else CLASS_NAMES[i]


def score(ck, rows, B, F, E, alpha, threshold, tau):
    ck["new_vegetable_head"]["switch_probability"] = tau
    idx, unk, _, _ = zs.decide(ck, B, F, E, alpha, threshold)
    labels = [r["class_name"] for r in rows]
    per_class = {}
    for c in dict.fromkeys(labels):
        m = [i for i, l in enumerate(labels) if l == c]
        if c in REJECT:
            good = int(unk[m].sum())
        elif c.startswith(("onion", "sweetpotato", "greenonion")):
            good = int(sum((idx[i] // 2 == CLASS_NAMES.index(c) // 2) and not unk[i] for i in m))
        else:
            good = int(sum(idx[i] == CLASS_NAMES.index(c) and not unk[i] for i in m))
        per_class[c] = {"n": len(m), "good": good, "answers": dict(collections.Counter(answer_name(idx[i], unk[i]) for i in m).most_common())}
    five = [i for i, l in enumerate(labels) if l in CLASS_NAMES[:10]]
    y5 = np.array([CLASS_NAMES.index(labels[i]) for i in five])
    kept = (idx[five] == y5) & ~unk[five]
    newv = [i for i, l in enumerate(labels) if l in CLASS_NAMES[10:]]
    yn = np.array([CLASS_NAMES.index(labels[i]) for i in newv])
    rej = [i for i, l in enumerate(labels) if l in REJECT]
    out = {
        "five_vegetables": {"n": len(five), "kept_correct": int(kept.sum()), "ci95": wilson(int(kept.sum()), len(five)),
                            "kind_right": int(((idx[five] // 2) == (y5 // 2)).sum()), "accuracy_ignoring_rejection": int((idx[five] == y5).sum()),
                            "turned_away": int(unk[five].sum()), "to_new_vegetable": int(((idx[five] >= 10) & ~unk[five]).sum())},
        "new_vegetables": {"n": len(newv), "kind_right_and_kept": int(((idx[newv] // 2 == yn // 2) & ~unk[newv]).sum()),
                           "class_right_and_kept": int(((idx[newv] == yn) & ~unk[newv]).sum()), "turned_away": int(unk[newv].sum())},
        "not_five": {"n": len(rej), "turned_away": int(unk[rej].sum()), "ci95": wilson(int(unk[rej].sum()), len(rej)),
                     **{k: f"{int(unk[[i for i in rej if labels[i] == k]].sum())}/{sum(labels[i] == k for i in rej)}" for k in REJECT}},
        "per_class": per_class,
        "answers": [answer_name(i, u) for i, u in zip(idx, unk)],
    }
    return out


def main():
    digest = seal_ok()
    rows = list(csv.DictReader(open(HERE / "t3_test.csv", encoding="utf-8")))
    res = {"seal": digest, "n": len(rows), "pipeline_check": pipeline_check()}
    F, B = features([str(IMG / f"{r['id']}.jpg") for r in rows])
    veg, other = zs.text_embeddings(); E = zs.evidence(F, veg, other)
    dev = json.load(open(REPO / "experiments/zero_shot_reject/dev_results.json"))
    ck = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    res["TV-11"] = t = score(ck, rows, B, F, E, 0.0, dev["0.0"]["threshold"], 0.8)
    res["ZS8"] = z = score(ck, rows, B, F, E, 0.25, dev["0.25"]["threshold"], 0.8)
    d = lambda a, b: z[a][b] - t[a][b]
    rules = {"five-vegetable kept >= -2": d("five_vegetables", "kept_correct") >= -2,
             "new-vegetable kind kept >= -1": d("new_vegetables", "kind_right_and_kept") >= -1,
             "not-five turned away >= +3": d("not_five", "turned_away") >= 3,
             "five vegetables to new vegetables <= +1": d("five_vegetables", "to_new_vegetable") <= 1}
    res["ZS8_rules"] = rules; res["ZS8_adopt"] = all(rules.values())
    res["differences"] = [{"id": r["id"], "class": r["class_name"], "TV-11": a, "ZS8": b}
                          for r, a, b in zip(rows, t["answers"], z["answers"]) if a != b]
    json.dump(res, open(HERE / "scores_once.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k not in ("TV-11", "ZS8")} | {
        s: {k: v for k, v in res[s].items() if k != "answers"} for s in ("TV-11", "ZS8")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
