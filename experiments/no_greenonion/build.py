"""experiments/no_greenonion (PREREG.md): G1 = TV-12 with the new-vegetable head retrained without green onion.

    python experiments/no_greenonion/build.py        # writes $DEPP_WORK/wm/G1.pth and heads_G1.pt here

- head_data.npz (the V7 head's stored CLIP features): the 97 green onion training rows get the label "not one of the
  vegetables" (column 16), so green onions are turned away; the 20 green onion rows of the new-vegetable Valid are left out of
  the threshold / temperature fitting (they should be turned away now).
- src/train_new_veg_head.py with tau 0.8 (as TV-12) and both green onion classes untrained.
Nothing else in model.pth changes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from labels import CLASS_NAMES  # noqa: E402

S = Path(os.environ["DEPP_WORK"])
W = S / "wm"
GREEN = [CLASS_NAMES.index("greenonion_fresh"), CLASS_NAMES.index("greenonion_rotten")]
OTHER = len(CLASS_NAMES)


def main():
    data = dict(np.load(REPO / "experiments/new_vegetables/v7/head_data.npz"))
    y = data["train_y"].astype(int)
    moved = int(np.isin(y, GREEN).sum())
    data["train_y"] = np.where(np.isin(y, GREEN), OTHER, y).astype(data["train_y"].dtype)
    keep = ~np.isin(data["dev_newval_y"].astype(int), GREEN)
    for key in ("dev_newval_X", "dev_newval_b", "dev_newval_y"):
        data[key] = data[key][keep]
    out_data = W / "head_data_G1.npz"
    np.savez(out_data, **data)
    print("green onion training rows relabelled:", moved, "| new-vegetable Valid rows kept for fitting:", int(keep.sum()), flush=True)
    out = W / "G1.pth"
    shutil.copy(REPO / "model.pth", out)
    subprocess.run([sys.executable, "src/train_new_veg_head.py", "--model", str(out), "--data", str(out_data), "--tau", "0.8",
                    "--untrained", "greenonion_fresh", "greenonion_rotten"], check=True, cwd=REPO)
    ck = torch.load(out, map_location="cpu", weights_only=True)
    ck["version"] = "G1 (experiments/no_greenonion): new-vegetable head without green onion"
    ck["new_vegetable_head"]["switch_rule"] = "tau 0.8 as in TV-12 (ZS8)"
    torch.save(ck, out)
    torch.save({k: ck[k] for k in ("new_vegetable_head", "version")}, HERE / "heads_G1.pt")
    print("saved", out)


if __name__ == "__main__":
    main()
