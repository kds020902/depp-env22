"""List of the new green onion photos for trial 1 (train+dev of the user's dataset + Codex photos Claude confirmed).

    python experiments/greenonion_condition/trial1/make_list.py

Paths are relative to D:/AI (the folder that holds both collections). Never lists the final test split.
Writes photos.csv (path, sha256, condition, group, origin) next to this file.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
GC = HERE.parent
USER = "green_onion_dataset/green_onion_dataset/final_dataset/"
CODEX = Path("greenonion_codex_collection_20261128")


def main():
    rows = []
    for r in csv.DictReader(open(GC / "user_lists.csv", encoding="utf-8")):
        if r["split"] == "test":
            continue
        rows.append({"path": USER + r["file"], "sha256": r["sha256"], "condition": r["condition"], "group": "u:" + r["source_id"], "origin": "user_" + r["split"]})
    review = list(csv.DictReader(open(Path("D:/AI") / CODEX / "claude_review.csv", encoding="utf-8")))
    for r in review:
        if r["claude_label"] not in ("spoiled", "fresh"):
            continue
        folder = "images_batch2" if r["candidate_id"].startswith("batch2") else "images"
        rel = f"{CODEX.as_posix()}/{folder}/{r['file']}"
        # fetch_pages.py saved page photos re-encoded as JPEG, so their recorded hash is of the downloaded bytes: hash the file itself
        sha = hashlib.sha256((Path("D:/AI") / rel).read_bytes()).hexdigest() if folder == "images_batch2" else r["sha256"]
        rows.append({"path": rel, "sha256": sha,
                     "condition": "spoiled" if r["claude_label"] == "spoiled" else "fresh", "group": "c:" + r["source_group"], "origin": "codex"})
    with open(HERE / "photos.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(Counter((r["origin"], r["condition"]) for r in rows), "groups", len({r["group"] for r in rows}))


if __name__ == "__main__":
    main()
