"""Blind test set T7 (RULES.md): real-scene tomatoes. Score TV-12 and C1 (TV-12 + the tomato crop check, t 0.9) once.

The photos stay in the working folder (licences); t7_test.csv lists them with source, licence and author. The script
refuses to run unless the SHA-256 of t7_test.csv equals the seal written at the end of RULES.md.

    python results/t7_test/score.py OUT.json
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "experiments/watermark"))
from common import S  # noqa: E402
from clip_model import ClipModel  # noqa: E402
from decision import apply_crop_check, decide, needs_crop_check  # noqa: E402
from images import five_crops, make_transform, open_image  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402
from networks import input_size, load_model  # noqa: E402

IMG = S / "t7/img"
AWAY = "거름"


def seal_ok() -> str:
    digest = hashlib.sha256((HERE / "t7_test.csv").read_bytes()).hexdigest()
    sealed = re.findall(r"SHA-256[^`]*`([0-9a-f]{64})`", (HERE / "RULES.md").read_text(encoding="utf-8"))
    assert sealed and sealed[-1] == digest, f"t7_test.csv does not match the seal in RULES.md ({digest})"
    return digest


def main():
    digest = seal_ok()
    rows = list(csv.DictReader(open(HERE / "t7_test.csv", encoding="utf-8")))
    paths = [str(IMG / f"{r['id']}.jpg") for r in rows]
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    c1 = copy.deepcopy(tv12); c1["tomato_crop_check"] = {"threshold": 0.9}
    model, ck = load_model(REPO / "model.pth", torch.device("cpu")); tf = make_transform(input_size(ck)); clip = ClipModel("cpu")
    B, F = [], []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            ims = [open_image(p) for p in paths[i:i + 32]]
            B.append(model(torch.stack([tf(im) for im in ims]))); F.append(clip.image_features(ims).cpu())
    b, f = torch.cat(B), torch.cat(F)
    res = {"seal": digest, "n": len(rows)}
    for name, checkpoint in (("TV-12", tv12), ("C1", c1)):
        a = decide(b, checkpoint, f)
        crops = torch.where(needs_crop_check(a, checkpoint))[0]
        if len(crops):
            with torch.inference_mode():
                fc = torch.stack([clip.image_features(five_crops(open_image(paths[i]))).cpu() for i in crops.tolist()])
            a = apply_crop_check(a, checkpoint, fc, crops)
        answers = [AWAY if u else CLASS_NAMES[int(i)] for i, u in zip(a.index, a.unknown)]
        count = lambda pred: sum(pred(r, x) for r, x in zip(rows, answers))
        res[name] = {
            "rotten_right": [count(lambda r, x: r["class_name"] == "tomato_rotten" and x == "tomato_rotten"), count(lambda r, x: r["class_name"] == "tomato_rotten")],
            "rotten_slight_right": [count(lambda r, x: r["severity"] == "slight" and x == "tomato_rotten"), count(lambda r, x: r["severity"] == "slight")],
            "rotten_heavy_right": [count(lambda r, x: r["severity"] == "heavy" and x == "tomato_rotten"), count(lambda r, x: r["severity"] == "heavy")],
            "fresh_right": [count(lambda r, x: r["class_name"] == "tomato_fresh" and x == "tomato_fresh"), count(lambda r, x: r["class_name"] == "tomato_fresh")],
            "rotten_to_fresh": count(lambda r, x: r["class_name"] == "tomato_rotten" and x == "tomato_fresh"),
            "fresh_to_rotten": count(lambda r, x: r["class_name"] == "tomato_fresh" and x == "tomato_rotten"),
            "turned_away": count(lambda r, x: x == AWAY), "other_vegetable": count(lambda r, x: x not in (AWAY, "tomato_fresh", "tomato_rotten")),
            "answers": answers}
    t, c = res["TV-12"], res["C1"]
    res["conditions"] = {"rotten tomato +3": c["rotten_right"][0] - t["rotten_right"][0] >= 3,
                         "fresh tomato >= -1": c["fresh_right"][0] - t["fresh_right"][0] >= -1}
    res["proposed"] = all(res["conditions"].values())
    res["differences"] = [{"id": r["id"], "class": r["class_name"], "severity": r["severity"], "TV-12": x, "C1": y}
                          for r, x, y in zip(rows, t["answers"], c["answers"]) if x != y]
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    for m in ("TV-12", "C1"):
        print(m, {k: v for k, v in res[m].items() if k != "answers"})
    print("conditions:", res["conditions"], "proposed:", res["proposed"], "differences:", len(res["differences"]))


if __name__ == "__main__":
    main()
