"""experiments/tomato_crops (PREREG.md): the tomato-only crop check on top of TV-12, on the development photos.

    python experiments/tomato_crops/evaluate.py OUT.json

- T3, T4, T5 (development now): photos and cached outputs of experiments/root_and_tomato/evaluate.py.
- Valid 300: dataset/ photos (stored outputs in head_data.npz for the whole-photo answer).
- Internet 221: the photos cannot be downloaded now (Flickr / Wikimedia rate limits); its tomato rows come from the recorded
  run of the same check with the same model (experiments/improve_without_photos/dev_crops.json), as PREREG.md says.
Crops are computed only for photos the system answers "fresh tomato" (cached in $DEPP_WORK/wm/tomato_crops.npz).
No blind test set is read.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("rt_eval", REPO / "experiments/root_and_tomato/evaluate.py")
rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
sys.path.insert(0, str(REPO / "src"))
from decision import apply_crop_check, decide, needs_crop_check  # noqa: E402
from images import five_crops, open_image  # noqa: E402

AWAY, CLASS_NAMES = rt.AWAY, rt.CLASS_NAMES
CACHE = rt.S / "wm/tomato_crops.npz"
THRESHOLDS = (0.6, 0.7, 0.8, 0.9)


def crop_features(paths: list[str]) -> np.ndarray:
    have = dict(np.load(CACHE)) if CACHE.exists() else {}
    todo = [p for p in paths if p not in have]
    if todo:
        from clip_model import ClipModel
        clip = ClipModel("cpu")
        with torch.inference_mode():
            for i in range(0, len(todo), 8):
                chunk = todo[i:i + 8]
                f = clip.image_features([c for p in chunk for c in five_crops(open_image(p))]).cpu().numpy().reshape(len(chunk), 5, -1)
                have.update(zip(chunk, f))
        np.savez(CACHE, **have)
    return np.stack([have[p] for p in paths]) if paths else np.zeros((0, 5, 512), np.float32)


def run(ck: dict, b, f, paths: list[str], t: float | None) -> list[str]:
    a = decide(torch.tensor(b), ck, torch.tensor(f))
    if t is not None:
        ck = {**ck, "tomato_crop_check": {"threshold": t}}
        rows = torch.where(needs_crop_check(a, ck))[0]
        a = apply_crop_check(a, ck, torch.tensor(crop_features([paths[i] for i in rows.tolist()])), rows)
    return [AWAY if u else CLASS_NAMES[int(i)] for i, u in zip(a.index, a.unknown)]


def main():
    sets = rt.photo_sets(); B, F = rt.outputs(sets)
    valid = [r for r in csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")) if r["split"] == "val"]
    paths = {n: sets[n][0] for n in ("t3", "t4", "t5")} | {"valid": [str(REPO / "dataset" / r["image_path"]) for r in valid]}
    assert [r["class_name"] for r in valid] == sets["valid"][1], "Valid order differs from head_data.npz"
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    recorded = json.load(open(REPO / "experiments/improve_without_photos/dev_crops.json"))
    res = {}
    for t in (None, *THRESHOLDS):
        ans = {n: run(tv12, B[n], F[n], paths[n], t) for n in paths}
        dev = [(y, a) for n in ("t3", "t4", "t5") for y, a in zip(sets[n][1], ans[n])]
        val = list(zip(sets["valid"][1], ans["valid"]))
        r = {"rotten_tomato_dev28": [sum(a == y for y, a in dev if y == "tomato_rotten"), sum(y == "tomato_rotten" for y, _ in dev)],
             "fresh_tomato_dev49": [sum(a == y for y, a in dev if y == "tomato_fresh"), sum(y == "tomato_fresh" for y, _ in dev)],
             "fresh_tomato_false_alarms_dev49_valid30": sum(y == "tomato_fresh" and a == "tomato_rotten" for y, a in dev + val),
             "valid_kept": sum(a == y for y, a in val), "valid_tomato": {c: sum(a == y for y, a in val if y == c) for c in ("tomato_fresh", "tomato_rotten")},
             "five_dev_right_and_kept": sum(a == y for y, a in dev if y in CLASS_NAMES[:10]),
             "answers": ans}
        if t is None:
            r["odev_kept_recorded"] = recorded["none"]["odev"]
        else:
            rec, base = recorded[f"t{t}"]["odev_per_class"], recorded["none"]["odev_per_class"]
            n = lambda d, c: int(d[c].split("/")[0])
            r["odev_kept_recorded"] = recorded["none"]["odev"] + (n(rec, "tomato_rotten") - n(base, "tomato_rotten")) - (n(base, "tomato_fresh") - n(rec, "tomato_fresh"))
            r["odev_tomato_recorded"] = {c: rec[c] for c in ("tomato_fresh", "tomato_rotten")}
        res["none" if t is None else f"t{t}"] = r
        print(t, {k: v for k, v in r.items() if k != "answers"}, flush=True)
    ok = [t for t in THRESHOLDS if res[f"t{t}"]["fresh_tomato_false_alarms_dev49_valid30"] <= 2]
    chosen = min(ok) if ok else None
    res["chosen_t"] = chosen
    if chosen is not None:
        c, b = res[f"t{chosen}"], res["none"]
        res["rules"] = {"rotten tomato +3 (bar)": c["rotten_tomato_dev28"][0] - b["rotten_tomato_dev28"][0] >= 3,
                        "fresh tomato >= -2": c["fresh_tomato_dev49"][0] - b["fresh_tomato_dev49"][0] >= -2,
                        "Valid kept >= -3": c["valid_kept"] - b["valid_kept"] >= -3,
                        "internet 221 kept >= -3 (recorded)": c["odev_kept_recorded"] - b["odev_kept_recorded"] >= -3}
        res["passed"] = all(res["rules"].values())
    json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
    print("chosen t:", chosen, res.get("rules"), "passed:", res.get("passed"))


if __name__ == "__main__":
    main()
