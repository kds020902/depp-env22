"""Blind test set T8 (RULES.md, addenda 1-3): score TV-12 and the potato candidate R3 once.

Photos stay in the working folder (copyright: private, non-commercial evaluation only); t8_test.csv lists them with
their source address. Refuses to run unless the SHA-256 of t8_test.csv equals the seal at the end of RULES.md.

    python results/t8_test/score.py OUT.json
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from clip_model import ClipModel  # noqa: E402
from decision import decide, held_back  # noqa: E402
from images import make_transform, open_image  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402
from networks import input_size, load_model  # noqa: E402

S = Path(os.environ["DEPP_WORK"])
IMG = S / "t8/img"
AWAY = "거름"
NEW = ("onion", "sweetpotato")


def seal_ok() -> str:
    digest = hashlib.sha256((HERE / "t8_test.csv").read_bytes()).hexdigest()
    sealed = re.findall(r"SHA-256[^`]*`([0-9a-f]{64})`", (HERE / "RULES.md").read_text(encoding="utf-8"))
    assert sealed and sealed[-1] == digest, f"t8_test.csv does not match the seal in RULES.md ({digest})"
    return digest


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def good(y: str, a: str) -> bool:
    if y == "rotten_fruit":
        return a == AWAY
    if y.startswith(NEW):
        return a != AWAY and a.rsplit("_", 1)[0] == y.rsplit("_", 1)[0]
    return a == y


def report(rows, answers, held) -> dict:
    out = {}
    for cls in sorted({r["class_name"] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r["class_name"] == cls]
        k = sum(good(cls, answers[i]) for i in idx)
        e = {"n": len(idx), "right": k, "rate": round(k / len(idx), 3), "ci95": wilson(k, len(idx)),
             "answers": dict(sorted({a: sum(answers[i] == a for i in idx) for a in {answers[i] for i in idx}}.items()))}
        if cls.endswith("_rotten"):
            e["called_fresh"] = sum(answers[i] == cls.replace("_rotten", "_fresh") for i in idx)
            for sev in ("heavy", "slight"):
                s = [i for i in idx if rows[i]["severity"] == sev]
                e[f"right_{sev}"] = f"{sum(good(cls, answers[i]) for i in s)}/{len(s)}"
        if cls.startswith(NEW):
            e["state_right"] = sum(answers[i] == cls for i in idx)
        out[cls] = e
    veg = [i for i, r in enumerate(rows) if r["class_name"] != "rotten_fruit"]
    rot = [i for i in veg if rows[i]["class_name"].endswith("_rotten")]
    k = sum(good(rows[i]["class_name"], answers[i]) for i in veg)
    out["_all_vegetables"] = {"n": len(veg), "right": k, "rate": round(k / len(veg), 3), "ci95": wilson(k, len(veg))}
    out["_rotten_called_fresh"] = {"n": len(rot), "count": sum(answers[i] == rows[i]["class_name"].replace("_rotten", "_fresh") for i in rot)}
    ans = [i for i in veg if not held[i] and answers[i] != AWAY]
    out["_hold_back_0.9"] = {"answered": len(ans), "answered_right": sum(good(rows[i]["class_name"], answers[i]) for i in ans),
                             "rotten_called_fresh_answered": sum(answers[i] == rows[i]["class_name"].replace("_rotten", "_fresh") for i in ans if i in rot)}
    return out


def main():
    digest = seal_ok()
    rows = list(csv.DictReader(open(HERE / "t8_test.csv", encoding="utf-8")))
    paths = [str(IMG / f"{r['id']}.jpg") for r in rows]
    model, ck = load_model(REPO / "model.pth", torch.device("cpu")); tf = make_transform(input_size(ck)); clip = ClipModel("cpu")
    B, F = [], []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            ims = [open_image(p) for p in paths[i:i + 32]]
            B.append(model(torch.stack([tf(im) for im in ims]))); F.append(clip.image_features(ims).cpu())
    b, f = torch.cat(B), torch.cat(F)
    models = {"TV-12": torch.load(REPO / "model.pth", map_location="cpu", weights_only=True),
              "R3": torch.load(S / "wm/R3.pth", map_location="cpu", weights_only=True)}
    clean = [i for i, r in enumerate(rows) if not r["flag"]]
    res = {"seal": digest, "n": len(rows), "n_clean": len(clean)}
    for name, c in models.items():
        a = decide(b, c, f); held = held_back(a).tolist()
        answers = [AWAY if u else CLASS_NAMES[int(i)] for i, u in zip(a.index, a.unknown)]
        res[name] = {"all": report(rows, answers, held),
                     "clean": report([rows[i] for i in clean], [answers[i] for i in clean], [held[i] for i in clean]),
                     "answers": answers}
    t, c = res["TV-12"]["clean"], res["R3"]["clean"]
    res["R3_conditions_clean"] = {   # experiments/root_and_tomato/PREREG.md addendum 3, written before sealing
        "rotten vegetables called fresh <= TV-12": c["_rotten_called_fresh"]["count"] <= t["_rotten_called_fresh"]["count"],
        "dirty fresh potato right >= TV-12 + 3": c["potato_fresh"]["right"] >= t["potato_fresh"]["right"] + 3,
        "rotten potato right >= TV-12 - 1": c["potato_rotten"]["right"] >= t["potato_rotten"]["right"] - 1,
        "rotten fruit turned away >= TV-12 - 2": c["rotten_fruit"]["right"] >= t["rotten_fruit"]["right"] - 2,
        "all vegetables right >= TV-12": c["_all_vegetables"]["right"] >= t["_all_vegetables"]["right"]}
    res["R3_proposed"] = all(res["R3_conditions_clean"].values())
    res["differences"] = [{"id": r["id"], "class": r["class_name"], "flag": r["flag"], "TV-12": x, "R3": y}
                          for r, x, y in zip(rows, res["TV-12"]["answers"], res["R3"]["answers"]) if x != y]
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    for name in models:
        for part in ("clean", "all"):
            r = res[name][part]
            print(name, part, {k: (v["right"], v["n"]) for k, v in r.items() if not k.startswith("_")}, r["_all_vegetables"], r["_rotten_called_fresh"])
    print(res["R3_conditions_clean"], "proposed:", res["R3_proposed"], "differences:", len(res["differences"]))


if __name__ == "__main__":
    main()
