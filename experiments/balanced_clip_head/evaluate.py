"""Development check of PREREG.md: TV-12 against B1 on the development photos only (no blind test set is touched).

    python experiments/balanced_clip_head/evaluate.py   -> dev_results.json

Same photos and the same 8 rules as experiments/t3_weak_spots (its round-2 development share, t5/lists/dev.csv), plus the
per-class "kept and right" counts on Valid, the first internet 221 and the 160 photos, so a class that loses can be seen.
B1 changes only the CLIP head, so the vegetable-model logits of TV-12 are reused.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
os.environ.setdefault("T3WS_LISTS", str(Path((os.environ.get("DEPP_WORK", "/tmp/claude-0/-home-user-depp-env22/a974ab14-e897-5da8-89f1-d4ea639bb37e/scratchpad") + "/t5/lists"))))
sys.path.insert(0, str(HERE.parent / "watermark"))
from common import REPO, W, clip_features, dev_sets, system  # noqa: E402
from model_utils import CLASS_NAMES  # noqa: E402

spec = importlib.util.spec_from_file_location("t3ws_evaluate", HERE.parent / "t3_weak_spots/evaluate.py")
t3ws = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3ws)


def per_class(ck) -> dict:
    sets = dev_sets(); F = clip_features(sets); B = dict(np.load(W / "b_WC.npz")); out = {}
    for name in ("valid", "odev", "o160", "asian37"):
        r = system(ck, B[name], F[name]); y = sets[name][1]
        out[name] = {CLASS_NAMES[c]: f"{int(((r['index'] == c) & ~r['unknown'])[y == c].sum())}/{int((y == c).sum())}"
                     for c in range(10) if (y == c).any()}
    return out


def main():
    res = {"TV-12": t3ws.score("TV-12", REPO / "model.pth", "WC"), "B1": t3ws.score("B1", W / "B1.pth", "WC")}
    for tag, path in (("TV-12", REPO / "model.pth"), ("B1", W / "B1.pth")):
        res[tag]["per_class"] = per_class(torch.load(path, map_location="cpu", weights_only=True))
    base, c = res["TV-12"], res["B1"]; d = lambda n, f: c[n][f] - base[n][f]
    checks = {"Valid kept >= -3": d("valid", "kept_correct") >= -3, "internet 221 kept >= -3": d("odev", "kept_correct") >= -3,
              "160 kept >= -3": d("o160", "kept_correct") >= -3, "Asian 37 kept >= -2": d("asian37", "kept_correct") >= -2,
              "everyday 400 turned away >= -5": d("coco400", "turned_away") >= -5,
              "new-vegetable Valid >= -2": d("newval", "kind_right_kept") >= -2,
              "old-vegetable dev to new <= +1": c["old_dev_to_new"] - base["old_dev_to_new"] <= 1,
              "pair classes >= -2": c["pair_total"] - base["pair_total"] >= -2}
    res["rules"] = {"checks": checks, "all_kept": all(checks.values()),
                    "note": "even if every rule holds, B1 is not adopted without a new blind test set (PREREG.md section 3)"}
    json.dump(res, open(HERE / "dev_results.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
