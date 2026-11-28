"""Model inputs of listed photos, so experiments run where the photos are not (the photos stay on the collector's computer).

    python experiments/sweetpotato_more/features_from_folder.py LIST.csv PHOTO_ROOT OUT.npz [--small FOLDER]

LIST.csv has a `path` column relative to PHOTO_ROOT (and sha256, which is checked). Every row whose file exists is stored:
ids = path, X = CLIP image features (the encoder of model.pth's heads), b = the vegetable model's logits.
--small FOLDER also writes a copy of each photo with its long side at most 640 px (JPEG), for looking at and for rot maps;
these copies are not what X and b were computed from.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from images import open_image  # noqa: E402
from predict import Classifier  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("list", type=Path); parser.add_argument("root", type=Path); parser.add_argument("out", type=Path)
    parser.add_argument("--small", type=Path)
    args = parser.parse_args()
    system = Classifier(REPO / "model.pth", torch.device("cpu"))
    ids, X, b, missing, changed = [], [], [], [], []
    for row in csv.DictReader(open(args.list, encoding="utf-8-sig")):
        path = args.root / row["path"]
        if not path.is_file():
            missing.append(row["path"]); continue
        if row.get("sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            changed.append(row["path"]); continue
        image = open_image(path)
        with torch.inference_mode():
            b.append(system.model(system.transform(image).unsqueeze(0))[0].numpy())
        X.append(system.clip.image_features([image])[0].numpy())
        ids.append(row["path"])
        if args.small:
            small = image.copy(); small.thumbnail((640, 640))
            target = (args.small / row["path"]).with_suffix(".jpg")
            target.parent.mkdir(parents=True, exist_ok=True)
            small.save(target, quality=85)
    # provenance, so a later run can check the cache was made from these exact photos with this model.pth
    sha = [hashlib.sha256((args.root / i).read_bytes()).hexdigest() for i in ids]
    model_sha = hashlib.sha256((REPO / "model.pth").read_bytes()).hexdigest()
    np.savez_compressed(args.out, ids=np.array(ids), X=np.array(X, np.float32), b=np.array(b, np.float32),
                        sha256=np.array(sha), model_sha256=np.array(model_sha))
    print(f"stored {len(ids)} photos; missing {len(missing)}; sha256 differs {len(changed)}")
    for name in missing + changed:
        print("  not stored:", name)


if __name__ == "__main__":
    main()
