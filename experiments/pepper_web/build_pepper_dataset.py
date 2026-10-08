"""Experiment (not adopted): add web photos of bell peppers to Train, 106 -> 140 fresh and 106 -> 140 rotten.

Valid and Test are not touched, so the 3rd-round Test 300 stays the same. The result is a separate folder for the
experiment; dataset/ itself was not changed.

Inputs
  --vetion   folder holding "Vegetables/" (Kaggle vetion/vege-quality v1, unzipped)
  --base     the 3rd-round dataset folder (manifest.csv + photos), e.g. dataset/ or the output of build_dataset_v3.py
  --review   experiments/pepper_web/visual_review.json (visual-review decisions)
  --out      output folder (a copy of --base plus the new photos)

Steps
  1. candidates = vetion photos that passed visual review; fresh and rotten come from the same source so the source
     is not a cue for either label (the rotten-only source that was also reviewed is not used)
  2. fill bell pepper Train up to 140 per class: all rotten that passed (34), fresh drawn at random (seed 0)
  3. stop if any of them duplicates a photo already in the dataset (the review already removed those)
  4. write train/<P_fresh|N_rotten>/ photos and the extended manifest.csv to --out
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "dataset" / "tools"))
from build_dataset_v2 import CONDITION_DIRS, T_DUP, T_SESSION, fingerprints, load_rgb, orb_inliers, union_find  # noqa: E402
from build_dataset_v3 import CLASS_NAMES, TRAIN_MAX, drop_old_duplicates, near_hash, similarity  # noqa: E402

SOURCE = ("vetion", "https://www.kaggle.com/datasets/vetion/vege-quality", "1")


def collect(args, base_rows):
    decisions = json.load(open(args.review, encoding="utf-8"))["decisions"]
    nodes = []
    for c in ("bellpepper_fresh", "bellpepper_rotten"):
        passed = sorted(decisions[f"vetion:{c}"]["passed"])
        room = TRAIN_MAX - sum(1 for r in base_rows if r["class_name"] == c and r["split"] == "train")
        if len(passed) < room:
            raise SystemExit(f"{c}: only {len(passed)} reviewed photos for {room} free Train places")
        for p in random.Random(0).sample(passed, room):
            path = args.vetion / p
            nodes.append(dict(source=SOURCE[0], class_name=c, archive_path=p, src_path=path, image=load_rgb(path)))
    return nodes


def groups(nodes, variants):
    """Copy groups and look-alike sessions among the new photos (v2 rules); all of them go to Train anyway."""
    S = similarity(variants)
    orb = orb_inliers(nodes, zip(*np.nonzero(np.triu(S >= 0.80, 1))))
    same = near_hash(nodes) | (S >= T_DUP)
    for (a, b), v in orb.items():
        if v >= 30 and S[a, b] >= 0.85:
            same[a, b] = same[b, a] = True
    dup = union_find(len(nodes), zip(*np.nonzero(np.triu(same, 1))))
    ses = union_find(len(nodes), zip(*np.nonzero(np.triu(same | (S >= T_SESSION), 1))))
    return dup, ses


def write(nodes, dup, ses, base_rows, args):
    out = args.out
    if out.resolve() == args.base.resolve():
        raise SystemExit("--out must be a new folder; dataset/ stays as it is.")
    for r in base_rows:
        dst = out / r["image_path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.base / r["image_path"], dst)
    order = sorted(range(len(nodes)), key=lambda i: (CLASS_NAMES.index(nodes[i]["class_name"]), nodes[i]["archive_path"]))
    g0 = max(int(r["group_id"][1:]) for r in base_rows)
    s0 = max(int(r["session_id"][1:]) for r in base_rows)
    next_id = collections.Counter({c: max(int(r["image_id"].rsplit("_", 1)[1]) for r in base_rows if r["class_name"] == c)
                                   for c in ("bellpepper_fresh", "bellpepper_rotten")})
    gmap, smap, rows = {}, {}, []
    for i in order:
        n, c = nodes[i], nodes[i]["class_name"]
        next_id[c] += 1
        image_id = f"{c}_{next_id[c]:04d}"
        dst = out / "train" / CONDITION_DIRS[c.split("_")[1]] / f"{image_id}{Path(n['archive_path']).suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(n["src_path"], dst)
        gmap.setdefault(int(dup[i]), f"g{g0 + len(gmap) + 1:04d}")
        smap.setdefault(int(ses[i]), f"s{s0 + len(smap) + 1:04d}")
        rows.append(dict(
            image_id=image_id, image_path=dst.relative_to(out).as_posix(), class_name=c, class_id=CLASS_NAMES.index(c),
            species="bellpepper", condition=c.split("_")[1],
            abnormal_type="none" if c.endswith("fresh") else "not_subtyped", split="train",
            group_id=gmap[int(dup[i])], session_id=smap[int(ses[i])], source_dataset=SOURCE[0],
            source_archive_path=n["archive_path"], source_url=SOURCE[1], source_version=SOURCE[2],
            label_origin="publisher_fresh_rotten_folder+visual_review", preprocessing="none",
            region_annotation="not_annotated", sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),
            pixel_sha256=n["pixel_sha256"], width=n["image"].width, height=n["image"].height, phash=n["phash"],
            dhash=n["dhash"], in_v1="no"))
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(base_rows[0]))
        writer.writeheader()
        writer.writerows(base_rows)
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vetion", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("dataset"))
    parser.add_argument("--review", type=Path, default=Path("experiments/pepper_web/visual_review.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    with open(args.base / "manifest.csv", newline="", encoding="utf-8-sig") as handle:
        base_rows = list(csv.DictReader(handle))
    if any(r["source_dataset"] == SOURCE[0] for r in base_rows):
        raise SystemExit("--base already holds these photos; start from the 3rd-round dataset.")
    nodes = collect(args, base_rows)
    old = [dict(source=r["source_dataset"], class_name=r["class_name"], archive_path=r["image_path"],
                src_path=args.base / r["image_path"], image=load_rgb(args.base / r["image_path"])) for r in base_rows]
    kept, variants, dropped = drop_old_duplicates(nodes, old)
    if dropped:
        raise SystemExit(f"{dropped} chosen photos duplicate photos already in the dataset; check the review file.")
    dup, ses = groups(kept, variants)
    rows = write(kept, dup, ses, base_rows, args)
    counts = collections.Counter(r["class_name"] for r in rows)
    print("added to train:", dict(counts))


if __name__ == "__main__":
    main()
