"""Breakdown of the adopted round-2 model (WC) next to V7, from the outputs already computed for the one final scoring
(final.py round2) and the development checks. No new photo set is scored; nothing here is used to choose anything.

    python experiments/watermark/breakdown_round2.py    -> breakdown_round2.json
"""

from __future__ import annotations

import csv
import json

import numpy as np
import torch

from common import REPO, S, W, dev_sets, system
from final import REJECT, photo_sets
from model_utils import CLASS_NAMES


def breakdown(p: np.ndarray, y: np.ndarray, unknown: np.ndarray | None = None) -> dict:
    right = p == y
    out = {"n": int(len(y)), "accuracy": int(right.sum()), "species": int((p // 2 == y // 2).sum()),
           "condition": int((p % 2 == y % 2).sum()), "rotten_found": int(((p % 2 == 1) & (y % 2 == 1)).sum()),
           "fresh_found": int(((p % 2 == 0) & (y % 2 == 0)).sum())}
    if unknown is not None:
        out.update(kept_correct=int((right & ~unknown).sum()), turned_away=int(unknown.sum()))
    return out


def five_only_any_photo(r, rows) -> int:
    good = 0
    for row, p, j in zip(rows, r["five"].argmax(1), r["five_unknown"]):
        c = row["category"]; lab = row["label"].split("|") if row["label"] else []; pred = CLASS_NAMES[p]
        if c in ("excluded_v2", "new_veg"):
            continue
        if c in REJECT: good += bool(j)
        elif c == "five_daily": good += (not j) and pred == lab[0]
        elif c == "cut_five": good += bool(j) or pred == lab[0]
        else: good += bool(j) or pred in lab
    return int(good)


def main():
    sealed = photo_sets(); F = dict(np.load(W / "final_clip.npz"))
    dev = dev_sets(); D = dict(np.load(W / "clip_cache.npz"))
    out = {}
    for tag in ("V7", "WC"):
        ck = torch.load(W / f"{tag}.pth", map_location="cpu", weights_only=True)
        B = dict(np.load(W / f"final_b_{tag}.npz")); Bd = dict(np.load(W / f"b_{tag}.npz"))
        res = {}
        y = np.array([CLASS_NAMES.index(c) for c in sealed["test_300"][1]])
        r = system(ck, B["test_300"], F["test_300"])
        res["test_system"] = breakdown(r["five"].argmax(1), y, r["five_unknown"])
        res["test_vegetable_model"] = breakdown(r["vegetable_model"].argmax(1), y)
        res["test_clip_head"] = breakdown(r["clip_head"].argmax(1), y)
        yv = dev["valid"][1]; rv = system(ck, Bd["valid"], D["valid"])
        res["valid_system"] = breakdown(rv["five"].argmax(1), yv, rv["five_unknown"])
        res["valid_vegetable_model"] = breakdown(rv["vegetable_model"].argmax(1), yv)
        y2 = np.array([CLASS_NAMES.index(c) for c in sealed["new_test_241"][1]])
        r2 = system(ck, B["new_test_241"], F["new_test_241"])
        res["new_test_241_five_only"] = breakdown(r2["five"].argmax(1), y2, r2["five_unknown"])
        yo = dev["odev"][1]; ro = system(ck, Bd["odev"], D["odev"])
        res["internet_221_five_only"] = breakdown(ro["five"].argmax(1), yo, ro["five_unknown"])
        res["internet_221_vegetable_model"] = breakdown(ro["vegetable_model"].argmax(1), yo)
        ra = system(ck, B["any_photo_v2"], F["any_photo_v2"])
        res["any_photo_five_only_good_of_247"] = five_only_any_photo(ra, sealed["any_photo_v2"][1])
        out[tag] = res
    json.dump(out, open(REPO / "experiments/watermark/breakdown_round2.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
