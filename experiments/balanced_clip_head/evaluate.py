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

import torch

HERE = Path(__file__).resolve().parent
os.environ.setdefault("T3WS_LISTS", os.environ["DEPP_WORK"] + "/t5/lists")
sys.path.insert(0, str(HERE.parent / "watermark"))
from common import REPO, W, kept_right_per_class  # noqa: E402

spec = importlib.util.spec_from_file_location("t3ws_evaluate", HERE.parent / "t3_weak_spots/evaluate.py")
t3ws = importlib.util.module_from_spec(spec); spec.loader.exec_module(t3ws)


def main():
    res = {"TV-12": t3ws.score("TV-12", REPO / "model.pth", "WC"), "B1": t3ws.score("B1", W / "B1.pth", "WC")}
    for tag, path in (("TV-12", REPO / "model.pth"), ("B1", W / "B1.pth")):
        res[tag]["per_class"] = kept_right_per_class(torch.load(path, map_location="cpu", weights_only=True))
    checks = t3ws.dev_rules(res["B1"], res["TV-12"])
    res["rules"] = {"checks": checks, "all_kept": all(checks.values()),
                    "note": "even if every rule holds, B1 is not adopted without a new blind test set (PREREG.md section 3)"}
    json.dump(res, open(HERE / "dev_results.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
