"""Checks run before any training in trial 3; any problem stops the run (exit 1).

    python experiments/greenonion_condition/trial3/integrity.py

1. The user's dataset still matches user_lists.csv (build_user_lists.py --check).
2. Every photo in photos.csv (user + Codex collections) exists under D:/AI and has the listed sha256.
3. features.npz has a row for every photo, was made from bytes with the same sha256, and with the current model.pth.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOT = Path("D:/AI")
DATASET = ROOT / "green_onion_dataset/green_onion_dataset/final_dataset"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def problems(photos: Path = HERE / "photos.csv", features: Path = HERE / "features.npz") -> list[str]:
    out = []
    r = subprocess.run([sys.executable, str(HERE.parent / "build_user_lists.py"), "--check", str(DATASET)], capture_output=True, text=True)
    if r.returncode:
        out.append("user dataset differs from user_lists.csv:\n" + r.stdout + r.stderr)
    rows = list(csv.DictReader(open(photos, encoding="utf-8")))
    for row in rows:
        p = ROOT / row["path"]
        if not p.is_file():
            out.append(f"missing photo: {row['path']}")
        elif sha(p) != row["sha256"]:
            out.append(f"photo changed: {row['path']}")
    if not features.is_file():
        return out + [f"no feature cache: {features}"]
    f = np.load(features)
    if "sha256" not in f.files or "model_sha256" not in f.files:
        return out + ["feature cache has no provenance (sha256 / model_sha256): remake it with features_from_folder.py"]
    if str(f["model_sha256"]) != sha(REPO / "model.pth"):
        out.append("feature cache was made with a different model.pth")
    cached = dict(zip(map(str, f["ids"]), map(str, f["sha256"])))
    for row in rows:
        if row["path"] not in cached:
            out.append(f"no features for: {row['path']}")
        elif cached[row["path"]] != row["sha256"]:
            out.append(f"features made from other bytes: {row['path']}")
    if any(r["split"] == "test" for r in rows if "split" in r) or any("test_holdout" in r["path"] for r in rows):
        out.append("photos.csv lists a final-test photo")
    return out


def main():
    p = problems()
    for line in p:
        print("  PROBLEM:", line)
    print("integrity OK" if not p else f"integrity FAILED ({len(p)} problems)")
    sys.exit(1 if p else 0)


if __name__ == "__main__":
    main()
