"""Lists of the user's green onion photos to use, from the files that actually remain on disk (not from the manifest).

    python experiments/greenonion_condition/build_user_lists.py DATASET_DIR            # (re)build the lists
    python experiments/greenonion_condition/build_user_lists.py --check DATASET_DIR    # exit 1 if a listed file is gone or changed

DATASET_DIR = .../green_onion_dataset/final_dataset. A photo is listed when its file exists and it is not in
user_dataset_review.csv with action 'exclude' or 'review' (fixed criteria set by eye before any model output).
Photos the user deleted are therefore never listed again, even though the manifest still names them.
Writes user_lists.csv (image_id, split, condition, file, sha256, source_id, environment) next to this file.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def check(root: Path) -> list[str]:
    """Problems between user_lists.csv and the files now on disk (run right before training; empty list = consistent)."""
    problems = []
    for r in csv.DictReader(open(HERE / "user_lists.csv", encoding="utf-8")):
        path = root / r["file"]
        if not path.exists():
            problems.append(f"missing: {r['image_id']} ({r['file']})")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != r["sha256"]:
            problems.append(f"changed: {r['image_id']} ({r['file']})")
    return problems


def main():
    if sys.argv[1] == "--check":
        problems = check(Path(sys.argv[2]))
        print("\n".join(problems) or "user_lists.csv matches the files")
        raise SystemExit(1 if problems else 0)
    root = Path(sys.argv[1])
    held = {r["image_id"]: r["action"] for r in csv.DictReader(open(HERE / "user_dataset_review.csv", encoding="utf-8"))}
    rows = list(csv.DictReader(open(root / "manifest_all.csv", encoding="utf-8-sig")))
    out, dropped = [], collections.Counter()
    for r in rows:
        path = root / r["local_rel_path"]
        if not path.exists():
            dropped["deleted by the user"] += 1; continue
        if r["image_id"] in held:
            dropped[f"listed as {held[r['image_id']]}"] += 1; continue
        out.append({"image_id": r["image_id"], "split": r["split"], "condition": "spoiled" if "상한" in r["label"] else "fresh",
                    "file": r["local_rel_path"], "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "source_id": r["source_id"], "environment": r["environment"]})
    with open(HERE / "user_lists.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
    print("kept", dict(collections.Counter((o["split"], o["condition"]) for o in out)))
    print("dropped", dict(dropped))


if __name__ == "__main__":
    main()
