"""experiments/sweetpotato_photos (PREREG.md): TV-12 with more rotten sweet potato photos in the new-vegetable head.

    python experiments/sweetpotato_photos/build.py PHOTOS.csv OUT.pth [--bank FEATURES.npz ...]

PHOTOS.csv: columns path,class_name (+ anything else); every row must be sweetpotato_rotten (the one thing changed).
The head is rebuilt by TV-12's own recipe (src/train_new_veg_head.py on experiments/new_vegetables/v7/head_data.npz, tau 0.8)
with the new rows appended to the training rows; nothing else in the checkpoint changes. With no rows the result is
TV-12's head exactly (checked on 2026-11-22: weights equal, 0 of 1,935 development answers changed).

Left out before training (image similarity only, no model answer is looked at): a photo whose CLIP cosine is >= 0.95 to a
development photo, to a training row, to a row of a --bank file (key "X" or "f": blind-test and dataset photos), or to an
earlier new photo. The kept rows' features go to added_features.npz, so the head can be rebuilt without the photos.
"""

from __future__ import annotations

import argparse
import csv
import json
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

HEAD_DATA = REPO / "experiments/new_vegetables/v7/head_data.npz"
ROOT_DEV = REPO / "experiments/sweetpotato_prompts/dev_features.npz"
ONLY = "sweetpotato_rotten"
TOO_CLOSE = 0.95


def image_features(paths: list[Path]) -> np.ndarray:
    from clip_model import ClipModel
    from images import open_image
    clip = ClipModel("cpu")
    return torch.cat([clip.image_features([open_image(p) for p in paths[i:i + 32]]).cpu() for i in range(0, len(paths), 32)]).numpy()


def too_close(new: np.ndarray, banks: dict[str, np.ndarray]) -> dict[int, list]:
    """index of a new photo -> [what it is too close to, cosine]; earlier new photos count as a bank."""
    out, x = {}, torch.tensor(new).float()
    for name, bank in banks.items():
        best = (x @ torch.tensor(bank).float().T).max(1).values
        for i in torch.where(best >= TOO_CLOSE)[0].tolist():
            out.setdefault(i, [name, round(float(best[i]), 4)])
    inner = x @ x.T
    for i in range(len(new)):
        if i not in out:
            for j in range(i):
                if j not in out and float(inner[i, j]) >= TOO_CLOSE:
                    out[i] = [f"new photo {j}", round(float(inner[i, j]), 4)]
                    break
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--bank", type=Path, nargs="*", default=[], help="More stored CLIP features to stay away from.")
    args = parser.parse_args()
    rows = list(csv.DictReader(open(args.photos, encoding="utf-8")))
    wrong = sorted({r["class_name"] for r in rows} - {ONLY})
    if wrong:
        raise SystemExit(f"only {ONLY} photos are added in this experiment; found {wrong}")
    data = dict(np.load(HEAD_DATA))
    kept, left_out = np.zeros((0, data["train_X"].shape[1]), np.float32), {}
    if rows:
        new = image_features([Path(r["path"]) for r in rows])
        banks = {"training row": data["train_X"], "potato / sweet potato development photo": np.load(ROOT_DEV)["X"],
                 **{f"development photo ({k[4:-2]})": data[k] for k in data if k.startswith("dev_") and k.endswith("_X")}}
        for path in args.bank:
            stored = np.load(path)
            banks[path.name] = stored["X"] if "X" in stored else stored["f"]
        left_out = too_close(new, banks)
        kept = new[[i for i in range(len(rows)) if i not in left_out]].astype(np.float32)
    data["train_X"] = np.concatenate([data["train_X"], kept])
    data["train_y"] = np.concatenate([data["train_y"], np.full(len(kept), CLASS_NAMES.index(ONLY), np.int16)])
    data["train_tag"] = np.concatenate([data["train_tag"], np.array(["sweetpotato_photos"] * len(kept), dtype=data["train_tag"].dtype)])
    work = args.out.with_suffix(".head_data.npz")
    np.savez(work, **data)
    shutil.copy(REPO / "model.pth", args.out)
    subprocess.run([sys.executable, "src/train_new_veg_head.py", "--model", str(args.out.resolve()), "--data", str(work.resolve()), "--tau", "0.8"],
                   check=True, cwd=REPO)
    if rows:
        np.savez_compressed(HERE / "added_features.npz", X=kept, paths=np.array([r["path"] for i, r in enumerate(rows) if i not in left_out]))
        json.dump({"given": len(rows), "kept": len(kept), "left_out": {rows[i]["path"]: why for i, why in left_out.items()}},
                  open(HERE / "dedup.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"added {len(kept)} of {len(rows)} photos ({len(left_out)} too close to an existing photo); saved {args.out}")


if __name__ == "__main__":
    main()
