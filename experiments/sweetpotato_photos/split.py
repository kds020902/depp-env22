"""experiments/sweetpotato_photos (PREREG.md, addendum 2): assign the collected photos to training or the blind test T9.

    python experiments/sweetpotato_photos/split.py            # reads collected.csv, writes split.json, prints the counts

Only rows with status "kept" are used (labels held back, near-duplicates and so on are already marked otherwise).
No photo and no model is read: the assignment depends on the list alone, so it can be made before any model sees a photo.

A source group (same photographer / session / dataset release, column source_group) goes to one side as a whole, over all
classes:
- a group with 3 or more rotten sweet potatoes is for training (T9 takes at most 2 photos of a class from one group);
- of the other groups with rotten sweet potatoes, those are taken for T9 whose rotten sweet potatoes add up to exactly
  T9_PER_CLASS: in the order of the SHA-256 of the group name, the earliest choice that can still be completed. The rest train;
- a group without rotten sweet potatoes is for T9.
Training uses the rotten sweet potatoes of its groups only; other photos of a training group are left out (they may not be
in T9, and this experiment adds no other class). In T9 each group gives at most 2 photos of a class, and each potato class
is cut to T9_PER_CLASS in the order (group hash, file name).
The experiment waits ("hold") unless T9 has exactly T9_PER_CLASS photos of each of the three classes and training has
at least TRAIN_MINIMUM rotten sweet potatoes.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ADDED, T9_CLASSES = "sweetpotato_rotten", ("sweetpotato_rotten", "potato_rotten", "potato_fresh")
T9_PER_CLASS, TRAIN_MINIMUM, TRAIN_GOAL, PER_GROUP_IN_T9 = 30, 20, 60, 2


def group_order(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def choose_exact(sizes: list[int], target: int) -> list[int] | None:
    """Indices of the earliest choice (taking an item whenever the target can still be met) that sums to `target`."""
    reachable = [{0}]                                   # reachable[i]: sums the items from i on can make (built from the end)
    for size in reversed(sizes):
        reachable.insert(0, reachable[0] | {s + size for s in reachable[0]})
    if target not in reachable[0]:
        return None
    chosen, left = [], target
    for i, size in enumerate(sizes):
        if size <= left and left - size in reachable[i + 1]:
            chosen.append(i); left -= size
    return chosen


def split(rows: list[dict]) -> dict:
    """rows: dicts with path, class_name, source_group. Returns the assignment of every row and the counts."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["source_group"]].append(r)
    names = sorted(groups, key=group_order)
    added = {g: sum(r["class_name"] == ADDED for r in groups[g]) for g in names}
    small = [g for g in names if 0 < added[g] <= PER_GROUP_IN_T9]
    picked = choose_exact([added[g] for g in small], T9_PER_CLASS)
    for_t9 = {small[i] for i in picked or []} | {g for g in names if added[g] == 0}
    assignment, in_t9 = {}, Counter()
    for g in names:
        taken = Counter()
        for r in sorted(groups[g], key=lambda r: r["path"]):
            c = r["class_name"]
            if g not in for_t9:
                assignment[r["path"]] = "train" if c == ADDED else "left out: its group is in training and only rotten sweet potatoes are added"
            elif c not in T9_CLASSES:
                assignment[r["path"]] = "left out: not a T9 class"
            elif taken[c] >= PER_GROUP_IN_T9:
                assignment[r["path"]] = f"left out: more than {PER_GROUP_IN_T9} photos of a class from one group"
            elif in_t9[c] >= T9_PER_CLASS:
                assignment[r["path"]] = f"left out: T9 already has {T9_PER_CLASS} of this class"
            else:
                assignment[r["path"]] = "t9"; taken[c] += 1; in_t9[c] += 1
    train = sum(a == "train" for a in assignment.values())
    reasons = [f"T9 has {in_t9[c]} of {T9_PER_CLASS} {c}" for c in T9_CLASSES if in_t9[c] != T9_PER_CLASS]
    if picked is None and sum(added[g] for g in small) >= T9_PER_CLASS:
        reasons.append(f"no choice of groups gives exactly {T9_PER_CLASS} rotten sweet potatoes for T9")
    if train < TRAIN_MINIMUM:
        reasons.append(f"training has {train} rotten sweet potatoes (minimum {TRAIN_MINIMUM}, goal {TRAIN_GOAL})")
    side = lambda g: "t9" if g in for_t9 else "train"
    return {"status": "hold" if reasons else "ready", "hold_reasons": reasons,
            "t9": {c: in_t9[c] for c in T9_CLASSES}, "train_rotten_sweet_potato": train,
            "groups": {"t9": sum(side(g) == "t9" for g in names), "train": sum(side(g) == "train" for g in names),
                       "t9_with_rotten_sweet_potato": len(picked or [])},
            "group_side": {g: side(g) for g in names}, "assignment": assignment}


def main():
    rows = list(csv.DictReader(open(HERE / "collected.csv", encoding="utf-8")))
    kept = [r for r in rows if r["status"] == "kept"]
    result = {"listed": len(rows), "not_kept": dict(Counter(r["status"] for r in rows if r["status"] != "kept")), **split(kept)}
    json.dump(result, open(HERE / "split.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print({k: v for k, v in result.items() if k not in ("assignment", "group_side")})


if __name__ == "__main__":
    main()
