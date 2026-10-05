"""Rebuild the 2nd-round dataset (1,200 photos, Train/Valid/Test 140/30/30 per class).

Inputs
  --src1         folder holding "Fruits_Vegetables_Dataset(12000)" (Kaggle muhriddinmuxiddinov, v2)
  --src2         folder holding "dataset/Train|Test/<freshcucumber|rottencucumber>" (Kaggle swoyam2609, v1)
  --v1-manifest  the 1st-round manifest.csv  (git show 03db425:manifest.csv > v1_manifest.csv)
  --review       data_review/v2_visual_review.json   (human visual-review decisions)
  --removed      data_review/v2_removed_from_v1.json (label-recheck removals)

Steps
  1. candidates = v1 photos (minus label-recheck removals) + photos that passed visual review
     (+30 fresh / 30 rotten cucumbers from the 2nd source, cropped out of its decorative frame)
  2. near-duplicate groups: pHash/dHash, flip/rotation-invariant ResNet18 cosine, ORB+RANSAC
  3. photo-session groups (looser cosine); big sessions never go to val/test
  4. per class: val/test 30 each from small groups, train 140 trimmed from the largest sessions,
     the mixed fresh/rotten tomato session balanced in train
  5. write images/ and manifest.csv to --out
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

import cv2
import imagehash
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

CLASS_NAMES = ["cucumber_fresh", "cucumber_rotten", "potato_fresh", "potato_rotten", "tomato_fresh", "tomato_rotten"]
SRC1_FOLDERS = {"cucumber_fresh": "FreshCucumber", "cucumber_rotten": "RottenCucumber", "potato_fresh": "FreshPotato",
                "potato_rotten": "RottenPotato", "tomato_fresh": "FreshTomato", "tomato_rotten": "RottenTomato"}
SOURCES = {"muhriddinmuxiddinov": ("https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset", "2"),
           "swoyam2609": ("https://www.kaggle.com/datasets/swoyam2609/fresh-and-stale-classification", "1")}
QUOTA = {"train": 140, "val": 30, "test": 30}
T_DUP, T_SESSION, BIG_SESSION, EVAL_SESSION_CAP = 0.92, 0.90, 30, 10


def load_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def frame_cropper(src2: Path):
    """The 2nd source pastes each photo into a fixed 450x320 frame; learn the frame and cut it away."""
    framed = sorted(p for p in (src2 / "dataset").rglob("*") if p.is_file() and "cucumber" in str(p))
    arrays = [np.asarray(load_rgb(p), dtype=np.int16) for p in framed]
    arrays = [a for a in arrays if a.shape[:2] == (320, 450)][::3]
    template = np.median(np.stack(arrays), axis=0)
    constant = np.stack(arrays).std(axis=0).mean(2) < 8

    def crop(path):
        image = load_rgb(path)
        a = np.asarray(image, dtype=np.int16)
        if a.shape[:2] == (320, 450):
            ys, xs = np.nonzero((np.abs(a - template).max(2) > 25) & ~constant)
            image = image.crop((int(np.percentile(xs, 0.5)), int(np.percentile(ys, 0.5)),
                                int(np.percentile(xs, 99.5)) + 1, int(np.percentile(ys, 99.5)) + 1))
        return image
    return crop


def collect_candidates(args):
    review = json.load(open(args.review, encoding="utf-8"))["decisions"]
    removed = json.load(open(args.removed, encoding="utf-8"))["removed"]
    label_fixed = {Path(r["image_path"]).stem for r in removed if r["reason"].startswith("label_recheck")}
    crop = frame_cropper(args.src2)
    nodes = []

    def add(source, class_name, archive_path, **extra):
        if source == "swoyam2609":
            image = crop(args.src2 / archive_path)
        else:
            image = load_rgb(args.src1 / archive_path)
        nodes.append(dict(source=source, class_name=class_name, archive_path=archive_path, image=image, **extra))

    with open(args.v1_manifest, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["image_id"] not in label_fixed:
                add("muhriddinmuxiddinov", row["class_name"], row["source_archive_path"],
                    old_id=row["image_id"], old_group=row["group_id"])
    for c in CLASS_NAMES:
        for p in review[c]["passed"]:
            add("muhriddinmuxiddinov", c, p)
    random.seed(2026)
    for c in ("cucumber_fresh", "cucumber_rotten"):
        for p in random.sample(review["src2_" + c]["passed"], 30):
            add("swoyam2609", c, p)
    # Extra review rounds aimed at photos unlike the big photo sessions (added in this order).
    have = {n["archive_path"] for n in nodes}
    rotten_x = review["tomato_rotten_x"]["passed"]
    first_round = [p for p in review["tomato_rotten_x"]["reviewed"][:36] if p in rotten_x]
    for c, paths in (("tomato_fresh", review["tomato_fresh_x"]["passed"]), ("tomato_rotten", first_round),
                     ("cucumber_fresh", review["cucumber_fresh_x"]["passed"]), ("tomato_rotten", rotten_x)):
        for p in paths:
            if p not in have:
                add("muhriddinmuxiddinov", c, p)
                have.add(p)
    return nodes


def fingerprints(nodes):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval()
    to_tensor = transforms.Compose([transforms.Resize((224, 224), antialias=True), transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    # 8 flips/rotations, so mirrored or rotated copies still look identical.
    ops = [lambda x: x, lambda x: x.flip(-1), lambda x: x.flip(-2), lambda x: x.flip(-1).flip(-2),
           lambda x: x.transpose(-1, -2), lambda x: x.transpose(-1, -2).flip(-1),
           lambda x: x.transpose(-1, -2).flip(-2), lambda x: x.transpose(-1, -2).flip(-1).flip(-2)]
    variants = np.zeros((len(nodes), 8, 512), dtype=np.float32)
    for start in range(0, len(nodes), 64):
        batch = torch.stack([to_tensor(n["image"]) for n in nodes[start:start + 64]])
        with torch.no_grad():
            for k, op in enumerate(ops):
                f = model(op(batch)).numpy()
                variants[start:start + len(batch), k] = f / np.linalg.norm(f, axis=1, keepdims=True)
    for n in nodes:
        n["phash"], n["dhash"] = str(imagehash.phash(n["image"])), str(imagehash.dhash(n["image"]))
        n["pixel_sha256"] = hashlib.sha256(n["image"].tobytes()).hexdigest()
    return variants


def orb_inliers(nodes, pairs):
    orb = cv2.ORB_create(nfeatures=800)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    feats = []
    for n in nodes:
        if n["source"] == "swoyam2609":
            gray = n["image"].convert("L")
        else:  # grayscale straight from the file, as when the dataset was built
            with Image.open(n["src_path"]) as raw:
                gray = ImageOps.exif_transpose(raw).convert("L")
        gray.thumbnail((480, 480))
        g = np.asarray(gray)
        fs = []
        for arr in (g, g[:, ::-1].copy()):  # also the mirror image
            kp, des = orb.detectAndCompute(arr, None)
            fs.append((np.float32([k.pt for k in kp]) if kp else None, des))
        feats.append(fs)

    def count(a, b):
        best = 0
        pa, da = feats[a][0]
        if da is None or len(da) < 10:
            return 0
        for pb, db in feats[b]:
            if db is None or len(db) < 10:
                continue
            good = [m for m, m2 in (x for x in matcher.knnMatch(da, db, k=2) if len(x) == 2)
                    if m.distance < 0.75 * m2.distance]
            if len(good) < 8:
                continue
            M, mask = cv2.estimateAffinePartial2D(pa[[m.queryIdx for m in good]], pb[[m.trainIdx for m in good]],
                                                  method=cv2.RANSAC, ransacReprojThreshold=5.0)
            if M is not None:
                best = max(best, int(mask.sum()))
        return best
    return {(int(a), int(b)): max(count(a, b), count(b, a)) for a, b in pairs}


def union_find(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        parent[find(a)] = find(b)
    return np.array([find(i) for i in range(n)])


def hamming(values):
    x = np.bitwise_xor(values[:, None], values[None, :])
    return np.unpackbits(x.view(np.uint8).reshape(len(values), len(values), 8), axis=2).sum(2)


def assign_splits(nodes, variants):
    N = len(nodes)
    E = variants[:, 0]
    S = np.max(np.stack([E @ variants[:, k].T for k in range(8)]), axis=0)
    S = np.maximum(S, S.T)
    near = (hamming(np.array([int(n["phash"], 16) for n in nodes], dtype=np.uint64)) <= 6) & \
           (hamming(np.array([int(n["dhash"], 16) for n in nodes], dtype=np.uint64)) <= 8)
    old = collections.defaultdict(list)
    for i, n in enumerate(nodes):
        if "old_group" in n:
            old[n["old_group"]].append(i)
    base = [(v[0], a) for v in old.values() for a in v[1:]]
    orb = orb_inliers(nodes, zip(*np.nonzero(np.triu(S >= 0.80, 1))))
    species = [n["class_name"].split("_")[0] for n in nodes]
    orb_edges = [(a, b) for (a, b), v in orb.items()
                 if (v >= 30 and S[a, b] >= 0.85) or (v >= 80 and species[a] == species[b])]
    dup = union_find(N, base + orb_edges + list(zip(*np.nonzero(np.triu(near | (S >= T_DUP), 1)))))
    ses = union_find(N, base + orb_edges + list(zip(*np.nonzero(np.triu(near | (S >= T_SESSION), 1)))))
    cls = [n["class_name"] for n in nodes]
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
                if all(cnt[(s, c)] + k <= QUOTA[s] for c, k in need.items()) and \
                   all(ses_eval[key] + k <= EVAL_SESSION_CAP for key, k in sc.items()):
                    dest = s
                    break
        for i in g:
            split[i] = dest
        if dest != "train":
            cnt.update({(dest, c): k for c, k in need.items()})
            ses_eval.update(sc)

    random.seed(7)  # trim train to quota, always from the class's largest session
    for c in sorted(set(cls)):
        tr = [i for i in range(N) if cls[i] == c and split[i] == "train"]
        random.shuffle(tr)
        while len(tr) > QUOTA["train"]:
            sizes = collections.Counter(int(ses[i]) for i in tr)
            big = max(sizes, key=lambda k: (sizes[k], k))
            victim = next((i for i in tr if int(ses[i]) == big and nodes[i]["source"] != "swoyam2609"), None)
            if victim is None:
                victim = next(i for i in tr if int(ses[i]) == big)
            tr.remove(victim)
            split[victim] = "drop"

    random.seed(11)  # a session holding both conditions of a species gets equal train counts
    balanced = set()
    for se in set(int(x) for x in ses):
        by = collections.defaultdict(list)
        for i in range(N):
            if int(ses[i]) == se and split[i] in ("train", "drop"):
                by[cls[i]].append(i)
        for sp in {c.split("_")[0] for c in by}:
            f, r = by.get(sp + "_fresh", []), by.get(sp + "_rotten", [])
            if len(f) < 10 or len(r) < 10 or min(len(f), len(r)) < 0.4 * max(len(f), len(r)):
                continue
            nf, nr = sum(split[i] == "train" for i in f), sum(split[i] == "train" for i in r)
            target = min(max(nf, nr), len(f), len(r))
            for grp, n in ((f, nf), (r, nr)):
                c = cls[grp[0]]
                if n < target:
                    ins = [i for i in grp if split[i] == "drop"][: target - n]
                    outs = [i for i in range(N) if cls[i] == c and split[i] == "train" and int(ses[i]) != se]
                    random.shuffle(outs)
                    outs.sort(key=lambda i: -sum(1 for j in range(N) if cls[j] == c and split[j] == "train"
                                                 and ses[j] == ses[i]))
                    for a, b in zip(ins, outs):
                        split[a], split[b] = "train", "drop"
                elif n > target:
                    for i in [i for i in grp if split[i] == "train"][: n - target]:
                        split[i] = "drop"
            balanced.add(se)
    for c in sorted(set(cls)):  # refill any class pushed below quota by the balancing
        while sum(1 for i in range(N) if cls[i] == c and split[i] == "train") < QUOTA["train"]:
            used = collections.Counter(int(ses[i]) for i in range(N) if cls[i] == c and split[i] == "train")
            pool = [i for i in range(N) if cls[i] == c and split[i] == "drop" and int(ses[i]) not in balanced]
            if not pool:
                raise RuntimeError(f"Not enough photos to fill train for {c}")
            split[min(pool, key=lambda i: (used[int(ses[i])], i))] = "train"
    return split, dup, ses


def write_dataset(nodes, split, dup, ses, out: Path):
    keep = sorted((i for i in range(len(nodes)) if split[i] != "drop"),
                  key=lambda i: (CLASS_NAMES.index(nodes[i]["class_name"]), nodes[i].get("old_id", "~"),
                                 nodes[i]["source"], nodes[i]["archive_path"]))
    gmap, smap, rows, next_id = {}, {}, [], 601
    for i in keep:
        n, c = nodes[i], nodes[i]["class_name"]
        if "old_id" in n:
            image_id, ext = n["old_id"], Path(n["archive_path"]).suffix.lower()
        else:
            image_id, next_id = f"{c}_{next_id:04d}", next_id + 1
            ext = ".png" if n["source"] == "swoyam2609" else Path(n["archive_path"]).suffix.lower()
        dst = out / "images" / c / f"{image_id}{ext}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if n["source"] == "swoyam2609":
            n["image"].save(dst)
        else:
            shutil.copy2(n["src_path"], dst)
        gmap.setdefault(int(dup[i]), f"g{len(gmap) + 1:04d}")
        smap.setdefault(int(ses[i]), f"s{len(smap) + 1:04d}")
        rows.append(dict(
            image_id=image_id, image_path=str(dst.relative_to(out)), class_name=c, class_id=CLASS_NAMES.index(c),
            species=c.split("_")[0], condition=c.split("_")[1],
            abnormal_type="none" if c.endswith("fresh") else "not_subtyped", split=split[i],
            group_id=gmap[int(dup[i])], session_id=smap[int(ses[i])], source_dataset=n["source"],
            source_archive_path=n["archive_path"], source_url=SOURCES[n["source"]][0],
            source_version=SOURCES[n["source"]][1], label_origin="publisher_fresh_rotten_folder+visual_review",
            preprocessing="frame_cropped" if n["source"] == "swoyam2609" else "none", region_annotation="not_annotated",
            sha256=hashlib.sha256(dst.read_bytes()).hexdigest(), pixel_sha256=n["pixel_sha256"],
            width=n["image"].width, height=n["image"].height, phash=n["phash"], dhash=n["dhash"],
            in_v1="yes" if "old_id" in n else "no"))
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src1", type=Path, required=True)
    parser.add_argument("--src2", type=Path, required=True)
    parser.add_argument("--v1-manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, default=Path("data_review/v2_visual_review.json"))
    parser.add_argument("--removed", type=Path, default=Path("data_review/v2_removed_from_v1.json"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    nodes = collect_candidates(args)
    for n in nodes:
        if n["source"] == "muhriddinmuxiddinov":
            n["src_path"] = args.src1 / n["archive_path"]
    print("candidates:", len(nodes), dict(collections.Counter(n["class_name"] for n in nodes)))
    split, dup, ses = assign_splits(nodes, fingerprints(nodes))
    rows = write_dataset(nodes, split, dup, ses, args.out)
    counts = collections.Counter((r["class_name"], r["split"]) for r in rows)
    for c in CLASS_NAMES:
        print(c, {s: counts[(c, s)] for s in QUOTA})


if __name__ == "__main__":
    main()
