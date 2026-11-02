"""Sealed test T4 (experiments/t3_weak_spots/PREREG.md): TV-12 and the candidate chosen on development photos, once each.
Also the earlier sealed tests once each for the candidate (TV-12 must give its recorded numbers: pipeline check).
Refuses to run unless t4_test.csv matches the seal written in PREREG.md.

    python results/t4_test/score.py   -> scores_once.json
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
sys.path.insert(0, str(REPO / "experiments/watermark"))
from common import S, W, system  # noqa: E402
from model_utils import CLASS_NAMES, input_size, load_model, make_transform, open_image  # noqa: E402

IMG = S / "t4/img"
WEAK = ("potato_fresh", "carrot_fresh", "tomato_rotten", "sweetpotato_fresh", "rotten_fruit")
PAIR = ("potato_rotten", "carrot_rotten", "tomato_fresh", "sweetpotato_rotten")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


wm = _load("wm_final", REPO / "experiments/watermark/final.py")


def seal_ok() -> str:
    digest = hashlib.sha256((HERE / "t4_test.csv").read_bytes()).hexdigest()
    text = (REPO / "experiments/t3_weak_spots/PREREG.md").read_text(encoding="utf-8")
    sealed = re.findall(r"t4_test\.csv SHA-256:\*\* `([0-9a-f]{64})`", text)
    assert sealed and sealed[-1] == digest, f"t4_test.csv does not match the seal ({digest})"
    return digest


def features(paths, model_path):
    from clip_model import ClipModel
    clip = ClipModel("cpu")
    model, ck = load_model(model_path, torch.device("cpu")); tf = make_transform(input_size(ck))
    F, B = [], []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            ims = [open_image(p) for p in paths[i:i + 32]]
            F.append(clip.image_features(ims).cpu()); B.append(model(torch.stack([tf(im) for im in ims])))
    return torch.cat(F).numpy(), torch.cat(B).numpy()


def t4_scores(ck, B, F, rows):
    r = system(ck, B, F); per = {}
    for c in dict.fromkeys(x["class_name"] for x in rows):
        m = np.array([x["class_name"] == c for x in rows])
        if c == "rotten_fruit":
            good = r["unknown"][m]
        elif c.startswith("sweetpotato"):
            good = (r["index"][m] // 2 == CLASS_NAMES.index(c) // 2) & ~r["unknown"][m]
        else:
            good = (r["index"][m] == CLASS_NAMES.index(c)) & ~r["unknown"][m]
        answers = {}
        for i, u in zip(r["index"][m], r["unknown"][m]):
            k = "거름" if u else CLASS_NAMES[i]; answers[k] = answers.get(k, 0) + 1
        per[c] = {"good": int(good.sum()), "n": int(m.sum()), "answers": answers}
    return {"per_class": per, "weak_total": sum(per[c]["good"] for c in WEAK if c in per),
            "pair_total": sum(per[c]["good"] for c in PAIR if c in per),
            "answers": ["거름" if u else CLASS_NAMES[i] for i, u in zip(r["index"], r["unknown"])]}


def earlier(ck, B, F, sets):
    out = {}
    for name, (paths, labels) in sets.items():
        r = system(ck, B[name], F[name])
        if name == "any_photo_v2":
            out[name] = wm.any_photo(r, labels)["good_old_categories"]; continue
        y = np.array([CLASS_NAMES.index(c) for c in labels])
        if name.startswith("new_veg_test"):
            out[name] = int(((r["index"] // 2 == y // 2) & ~r["unknown"]).sum()); continue
        out[name] = {"accuracy": int((r["index"] == y).sum()), "kept_correct": int(((r["index"] == y) & ~r["unknown"]).sum())}
    out["new_veg_tests_total"] = sum(out[f"new_veg_test{k}"] for k in range(1, 5))
    return out


def main():
    digest = seal_ok()
    dev = json.load(open(REPO / "experiments/t3_weak_spots/dev_results.json"))
    tag = dev["chosen"]; assert tag, "no candidate was chosen on the development photos"
    rows = list(csv.DictReader(open(HERE / "t4_test.csv", encoding="utf-8")))
    paths = [str(IMG / f"{r['id']}.jpg") for r in rows]
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    cand_path = W / f"{tag}.pth"; cand = torch.load(cand_path, map_location="cpu", weights_only=True)
    F, B12 = features(paths, REPO / "model.pth")
    Bc = B12 if tag in ("K1", "K3") else features(paths, cand_path)[1]
    res = {"seal": digest, "n": len(rows), "candidate": tag, "TV-12": t4_scores(tv12, B12, F, rows), tag: t4_scores(cand, Bc, F, rows)}
    sets = wm.photo_sets(); Fe = dict(np.load(W / "final_clip.npz"))
    Be12 = dict(np.load(W / "final_b_WC.npz"))
    Bec = Be12 if tag in ("K1", "K3") else wm.logits(tag, sets)  # caches final_b_<tag>.npz
    res["earlier_TV-12"] = e12 = earlier(tv12, Be12, Fe, sets)
    assert (e12["test_300"]["kept_correct"], e12["new_test_241"]["kept_correct"], e12["any_photo_v2"], e12["new_veg_tests_total"]) == (285, 188, 216, 176), e12
    res[f"earlier_{tag}"] = ec = earlier(cand, Bec, Fe, sets)
    t, c = res["TV-12"], res[tag]
    rules = {"T4 weak classes >= +8": c["weak_total"] - t["weak_total"] >= 8,
             "T4 pair classes >= -2": c["pair_total"] - t["pair_total"] >= -2,
             "Test 300 kept >= -3": ec["test_300"]["kept_correct"] - e12["test_300"]["kept_correct"] >= -3,
             "internet 241 kept >= -3": ec["new_test_241"]["kept_correct"] - e12["new_test_241"]["kept_correct"] >= -3,
             "any-photo good >= -3": ec["any_photo_v2"] - e12["any_photo_v2"] >= -3,
             "new-vegetable tests >= -3": ec["new_veg_tests_total"] - e12["new_veg_tests_total"] >= -3}
    res["rules"] = rules; res["adopt"] = all(rules.values())
    res["differences"] = [{"id": r["id"], "class": r["class_name"], "TV-12": a, tag: b}
                          for r, a, b in zip(rows, t["answers"], c["answers"]) if a != b]
    json.dump(res, open(HERE / "scores_once.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in res.items() if k not in ("TV-12", tag)} | {
        s: {k: v for k, v in res[s].items() if k != "answers"} for s in ("TV-12", tag)}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
