"""Set aside T10 (a new blind test) from the new sweet potato photos BEFORE any of them is used.

    python experiments/t10_split/split.py

Rule (fixed before looking at any model output on these photos):
- unit = source_group (one iNaturalist user = one group), so one person's photos never sit on both sides
- a group goes to T10 when int(sha256("T10:" + source_group)[:8], 16) % 5 < 2  (about 40 % of groups)
- only rows with status == kept; the class label is the list's class_name
Writes t10_ids.txt / train_ids.txt (sorted ids) and counts.json. Codex's list joins later with the same rule.
"""

from __future__ import annotations

import collections
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
LISTS = [HERE.parents[0] / "sweetpotato_more" / "collected.csv"]


def to_t10(group: str) -> bool:
    return int(hashlib.sha256(("T10:" + group).encode("utf-8")).hexdigest()[:8], 16) % 5 < 2


def main():
    t10, train = [], []
    count = collections.Counter()
    for path in LISTS:
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["status"] != "kept":
                    continue
                side = "T10" if to_t10(row["source_group"]) else "train"
                (t10 if side == "T10" else train).append(row["id"])
                count[(side, row["class_name"])] += 1
    (HERE / "t10_ids.txt").write_text("\n".join(sorted(t10)) + "\n", encoding="utf-8")
    (HERE / "train_ids.txt").write_text("\n".join(sorted(train)) + "\n", encoding="utf-8")
    out = {f"{s}/{c}": n for (s, c), n in sorted(count.items())}
    (HERE / "counts.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
