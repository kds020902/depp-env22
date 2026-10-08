"""Add bell pepper and carrot (fresh / rotten) to the 2nd-round dataset: 6 classes -> 10 classes.

The 1,200 photos of the 2nd round keep their split, id and folder; only new photos are added.

Inputs
  --src1     folder holding "Fruits_Vegetables_Dataset(12000)" (Kaggle muhriddinmuxiddinov, v2)
  --base     the 2nd-round dataset folder (manifest.csv + photos), e.g. dataset/ or the output of build_dataset_v2.py
  --review   dataset/review/v3_visual_review.json (visual-review decisions for the new classes)
  --same     dataset/review/v3_same_photo_pairs.json (copies judged by eye; thumbnails the rules below miss)
  --out      output folder; may be the same as --base (photos are added next to the existing ones)

Steps
  1. candidates = photos that passed visual review; drop any that duplicate a photo already in the dataset
  2. near-duplicate groups with the v2 rule (pHash/dHash, flip/rotation-invariant ResNet18 cosine, ORB+RANSAC)
     plus the pairs judged by eye to be the same photo
  3. sessions: the publisher's own photo shoot (every 224x224 photo of a fresh class) is one session and stays in
     train; the other photos form sessions with the v2 rule (cosine 0.90). Big sessions never go to val/test.
  4. per class: val/test 30 each from small groups; train = what the scarcer condition of the vegetable allows
     (at most 140), trimmed from the largest sessions, so fresh and rotten keep the same counts in every split
  5. write <train|valid|test>/<P_fresh|N_rotten>/ photos and the extended manifest.csv to --out
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dataset_v2 import (  # noqa: E402
    BIG_SESSION, CONDITION_DIRS, EVAL_SESSION_CAP, SOURCES, SPLIT_DIRS, T_DUP, T_SESSION,
    fingerprints, hamming, load_rgb, orb_inliers, union_find,
)

OLD_CLASSES = ["cucumber_fresh", "cucumber_rotten", "potato_fresh", "potato_rotten", "tomato_fresh", "tomato_rotten"]
NEW_CLASSES = ["bellpepper_fresh", "bellpepper_rotten", "carrot_fresh", "carrot_rotten"]
CLASS_NAMES = OLD_CLASSES + NEW_CLASSES
EVAL_QUOTA, TRAIN_MAX = 30, 140
SHOOT_SIZE = (224, 224)  # the publisher resized its own market photos to 224x224


def collect(args):
    decisions = json.load(open(args.review, encoding="utf-8"))["decisions"]
    nodes = []
    for c in NEW_CLASSES:
        for key in (c, c + "_shoot", c + "_shoot2"):
            for p in decisions.get(key, {}).get("passed", []):
                path = args.src1 / p
                nodes.append(dict(source="muhriddinmuxiddinov", class_name=c, archive_path=p, src_path=path,
                                  image=load_rgb(path)))
    with open(args.base / "manifest.csv", newline="", encoding="utf-8-sig") as handle:
        base_rows = [r for r in csv.DictReader(handle) if r["class_name"] in OLD_CLASSES]
    old = [dict(source="muhriddinmuxiddinov", class_name=r["class_name"], archive_path=r["image_path"],
                src_path=args.base / r["image_path"], image=load_rgb(args.base / r["image_path"])) for r in base_rows]
    return nodes, old, base_rows


def similarity(variants):
    E = variants[:, 0]
    S = np.max(np.stack([E @ variants[:, k].T for k in range(8)]), axis=0)
    return np.maximum(S, S.T)


def near_hash(nodes):
    return (hamming(np.array([int(n["phash"], 16) for n in nodes], dtype=np.uint64)) <= 6) & \
           (hamming(np.array([int(n["dhash"], 16) for n in nodes], dtype=np.uint64)) <= 8)


def drop_old_duplicates(nodes, old):
    """A candidate that matches any photo already in the dataset is dropped (it could cross splits)."""
    both = nodes + old
    variants = fingerprints(both)
    S = similarity(variants)
    near = near_hash(both)
    n = len(nodes)
    cand = (near | (S >= T_DUP))[:n, n:]
    pairs = [(a, n + b) for a, b in zip(*np.nonzero(S[:n, n:] >= 0.80))]
    orb = orb_inliers(both, pairs)
    for (a, b), v in orb.items():
        if v >= 30 and S[a, b] >= 0.85:
            cand[a, b - n] = True
    keep = ~cand.any(1)
    return [nd for nd, k in zip(nodes, keep) if k], variants[:n][keep], int((~keep).sum())


def assign(nodes, variants, same_pairs):
    N = len(nodes)
    S = similarity(variants)
    near = near_hash(nodes)
    orb = orb_inliers(nodes, zip(*np.nonzero(np.triu(S >= 0.80, 1))))
    species = [n["class_name"].split("_")[0] for n in nodes]
    orb_edges = [(a, b) for (a, b), v in orb.items()
                 if (v >= 30 and S[a, b] >= 0.85) or (v >= 80 and species[a] == species[b])]
    index = {n["archive_path"]: i for i, n in enumerate(nodes)}
    same_edges = [(index[a], index[b]) for a, b in same_pairs if a in index and b in index]
    dup = union_find(N, same_edges + orb_edges + list(zip(*np.nonzero(np.triu(near | (S >= T_DUP), 1)))))
    shoot = np.array([n["class_name"].endswith("fresh") and n["image"].size == SHOOT_SIZE for n in nodes])
    # Similar-looking fresh peppers/carrots chain into one huge session at cosine 0.90, so sessions are built
    # from the web photos only; the publisher's shoot is one session per class.
    web = ~shoot
    ses_edges = [(a, b) for a, b in same_edges + orb_edges if web[a] and web[b]]
    ses_edges += [(a, b) for a, b in zip(*np.nonzero(np.triu(near | (S >= T_SESSION), 1))) if web[a] and web[b]]
    ses = union_find(N, ses_edges)
    cls = [n["class_name"] for n in nodes]
    for c in set(cls):
        members = [i for i in range(N) if shoot[i] and cls[i] == c]
        for i in members:
            ses[i] = members[0]
    ses_size = collections.Counter(int(x) for x in ses)

    groups = collections.defaultdict(list)
    for i in range(N):
        groups[int(dup[i])].append(i)
    order = list(groups.values())
    random.seed(42)
    random.shuffle(order)
    order.sort(key=len)  # smallest groups first: val/test get the most varied photos
    split, cnt, ses_eval = {}, collections.Counter(), collections.Counter()
    for g in order:
        need = collections.Counter(cls[i] for i in g)
        sc = collections.Counter((cls[i], int(ses[i])) for i in g)
        dest = "train"
        if not any(ses_size[int(ses[i])] >= BIG_SESSION for i in g):
            for s in ("test", "val"):
                if all(cnt[(s, c)] + k <= EVAL_QUOTA for c, k in need.items()) and \
                   all(ses_eval[key] + k <= EVAL_SESSION_CAP for key, k in sc.items()):
                    dest = s
                    break
        for i in g:
            split[i] = dest
        if dest != "train":
            cnt.update({(dest, c): k for c, k in need.items()})
            ses_eval.update(sc)
    for c in sorted(set(cls)):
        for s in ("val", "test"):
            if cnt[(s, c)] != EVAL_QUOTA:
                raise RuntimeError(f"Only {cnt[(s, c)]} {s} photos for {c}")

    # Fresh and rotten of one vegetable get the same train count: what the scarcer one has (at most 140).
    train_quota = {}
    for sp in sorted(set(species)):
        have = [sum(1 for i in range(N) if cls[i] == f"{sp}_{cond}" and split[i] == "train") for cond in ("fresh", "rotten")]
        train_quota[f"{sp}_fresh"] = train_quota[f"{sp}_rotten"] = min(min(have), TRAIN_MAX)
    random.seed(7)  # trim train to quota, always from the class's largest session
    for c in sorted(set(cls)):
        tr = [i for i in range(N) if cls[i] == c and split[i] == "train"]
        random.shuffle(tr)
        while len(tr) > train_quota[c]:
            sizes = collections.Counter(int(ses[i]) for i in tr)
            big = max(sizes, key=lambda k: (sizes[k], k))
            victim = next(i for i in tr if int(ses[i]) == big)
            tr.remove(victim)
            split[victim] = "drop"
    return split, dup, ses, shoot, train_quota


def write(nodes, split, dup, ses, shoot, base_rows, args):
    out = args.out
    if out.resolve() != args.base.resolve():
        for r in base_rows:
            dst = out / r["image_path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.base / r["image_path"], dst)
    keep = sorted((i for i in range(len(nodes)) if split[i] != "drop"),
                  key=lambda i: (CLASS_NAMES.index(nodes[i]["class_name"]), nodes[i]["archive_path"]))
    g0 = max(int(r["group_id"][1:]) for r in base_rows)
    s0 = max(int(r["session_id"][1:]) for r in base_rows)
    gmap, smap, rows, next_id = {}, {}, [], collections.Counter()
    for i in keep:
        n, c = nodes[i], nodes[i]["class_name"]
        next_id[c] += 1
        image_id = f"{c}_{next_id[c]:04d}"
        dst = out / SPLIT_DIRS[split[i]] / CONDITION_DIRS[c.split("_")[1]] / f"{image_id}{Path(n['archive_path']).suffix.lower()}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(n["src_path"], dst)
        gmap.setdefault(int(dup[i]), f"g{g0 + len(gmap) + 1:04d}")
        smap.setdefault(int(ses[i]), f"s{s0 + len(smap) + 1:04d}")
        rows.append(dict(
            image_id=image_id, image_path=str(dst.relative_to(out)), class_name=c, class_id=CLASS_NAMES.index(c),
            species=c.split("_")[0], condition=c.split("_")[1],
            abnormal_type="none" if c.endswith("fresh") else "not_subtyped", split=split[i],
            group_id=gmap[int(dup[i])], session_id=smap[int(ses[i])], source_dataset=n["source"],
            source_archive_path=n["archive_path"], source_url=SOURCES[n["source"]][0],
            source_version=SOURCES[n["source"]][1],
            label_origin="publisher_fresh_rotten_folder+visual_review" + ("+publisher_shoot" if shoot[i] else ""),
            preprocessing="none", region_annotation="not_annotated",
            sha256=hashlib.sha256(dst.read_bytes()).hexdigest(), pixel_sha256=n["pixel_sha256"],
            width=n["image"].width, height=n["image"].height, phash=n["phash"], dhash=n["dhash"], in_v1="no"))
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(base_rows[0]))
        writer.writeheader()
        writer.writerows(base_rows)
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src1", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=Path("dataset"))
    parser.add_argument("--review", type=Path, default=Path("dataset/review/v3_visual_review.json"))
    parser.add_argument("--same", type=Path, default=Path("dataset/review/v3_same_photo_pairs.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    if any(r["class_name"] in NEW_CLASSES for r in csv.DictReader(open(args.base / "manifest.csv", encoding="utf-8"))):
        raise SystemExit("--base already holds bell pepper/carrot photos; start from the 2nd-round dataset.")
    nodes, old, base_rows = collect(args)
    print("candidates:", len(nodes), dict(collections.Counter(n["class_name"] for n in nodes)))
    nodes, variants, dropped = drop_old_duplicates(nodes, old)
    print("dropped as duplicates of existing photos:", dropped)
    split, dup, ses, shoot, quota = assign(nodes, variants, json.load(open(args.same, encoding="utf-8"))["pairs"])
    rows = write(nodes, split, dup, ses, shoot, base_rows, args)
    counts = collections.Counter((r["class_name"], r["split"]) for r in rows)
    for c in NEW_CLASSES:
        print(c, {s: counts[(c, s)] for s in ("train", "val", "test")})


if __name__ == "__main__":
    main()
