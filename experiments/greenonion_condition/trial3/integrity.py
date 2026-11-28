"""Checks run before any training in trial 3; any problem stops the run (exit 1).

    python experiments/greenonion_condition/trial3/integrity.py

1. The user's dataset still matches user_lists.csv (build_user_lists.py --check).
2. Codex's verify_trial_inputs.py (copied unchanged from D:/AI/greenonion_codex_review): every photo in photos.csv
   exists and has its sha256, no final-test photo is listed, and features.npz was made from exactly this list,
   these photo bytes, this model.pth and this extractor (features_from_folder.py).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from verify_trial_inputs import verify  # noqa: E402

ROOT = Path("D:/AI")
DATASET = ROOT / "green_onion_dataset/green_onion_dataset/final_dataset"
EXTRACTOR = REPO / "experiments/sweetpotato_more/features_from_folder.py"


def problems(photos: Path = HERE / "photos.csv", features: Path = HERE / "features.npz") -> list[str]:
    out = []
    r = subprocess.run([sys.executable, str(HERE.parent / "build_user_lists.py"), "--check", str(DATASET)], capture_output=True, text=True)
    if r.returncode:
        out.append("user dataset differs from user_lists.csv:\n" + r.stdout + r.stderr)
    try:
        out += verify(photos, ROOT, features, REPO / "model.pth", EXTRACTOR)["errors"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        out.append(f"{type(e).__name__}: {e}")
    return out


def main():
    p = problems()
    for line in p:
        print("  PROBLEM:", line)
    print("integrity OK" if not p else f"integrity FAILED ({len(p)} problems)")
    sys.exit(1 if p else 0)


if __name__ == "__main__":
    main()
