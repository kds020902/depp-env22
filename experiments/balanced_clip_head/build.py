"""Build candidate B1 of PREREG.md: the CLIP head retrained on a balanced list (fresh : rotten = 1 : 1 per vegetable).

    python experiments/balanced_clip_head/build.py lists   # balanced list -> working folder t6/b1/, b1_counts.json here
    python experiments/balanced_clip_head/build.py B1      # CLIP head on that list with TV-12's rules -> wm/B1.pth

The photos stay in the working folder (licences); the review labels are the record files of the working folder
(t6/collapse_labels.json, t6/local_labels.json, t6/web*/labels.json). Rules (PREREG.md):
- collapsed rotten photos (shape gone) are left out;
- per vegetable T = max(350, smaller side of TV-12's list); the larger side is cut at random (seed 42), Asian fresh
  cucumbers are all kept (addendum 1); the smaller side is filled in this order: weak-spot training share, local
  datasets, internet; if a side stays short, T is what was found and the other side is cut to it;
- an added photo is skipped when its CLIP cosine with TV-12's training photos, any evaluation photo or an earlier
  added photo is 0.95 or more;
- the CLIP head is trained exactly like TV-12's (src/train_clip_head.py, light watermark copies, 2,850 not-five photos),
  then TV-12's text evidence (alpha 0.25) is put back and the not-five threshold refitted (95% of Valid + 192 LVIS crops).
  The vegetable model and the new-vegetable head are TV-12's.
"""

from __future__ import annotations

import csv
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "watermark"))
sys.path.insert(0, str(HERE.parent / "t3_weak_spots"))
from common import REPO, S, W  # noqa: E402
from model_utils import CLASS_NAMES  # noqa: E402

T6 = S / "t6"
OUT = T6 / "b1"
SPECIES = ("tomato", "potato", "cucumber", "bellpepper", "carrot")
FIVE = CLASS_NAMES[:10]
MIN_T = 350
COPY = 0.95
CACHE = W / "head_features.npz"  # the CLIP feature cache of TV-12's head (same keys as src/train_clip_head.py)


def base_rows() -> list[dict]:
    """TV-12's five-vegetable training photos of the CLIP head: manifest Train + the five-vegetable rows of the extra CSV."""
    man = [r for r in csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")) if r["split"] == "train"]
    ext = [r for r in csv.DictReader(open(S / "negx/extra_train_N3.csv", encoding="utf-8")) if r["class_name"] in FIVE]
    return ([{"path": str(REPO / "dataset" / r["image_path"]), "class_name": r["class_name"], "source": "manifest_train",
              "group": r["source_dataset"]} for r in man]
            + [{"path": r["path"], "class_name": r["class_name"], "source": "extra_train_N3", "group": r["group"]} for r in ext])


def fill_pools() -> dict[str, list[dict]]:
    """class -> reviewed photos that may be added, in the order of PREREG.md (weak-spot share, local, internet)."""
    pools = {c: [] for c in FIVE}
    for r in csv.DictReader(open(S / "t5/lists/train.csv", encoding="utf-8")):
        if r["class_name"] in FIVE:
            pools[r["class_name"]].append({"path": r["path"], "class_name": r["class_name"], "source": "weak_spot_share", "group": r["user"]})
    lab = json.load(open(T6 / "local_labels.json"))
    for r in json.load(open(T6 / "local_candidates.json")):
        if lab.get(r["path"]) == "ok":
            pools[r["class_name"]].append({"path": r["path"], "class_name": r["class_name"], "source": "local_dataset", "group": r["src"]})
    for web in ("web", "web2"):
        f = T6 / web / "labels.json"
        if f.exists():
            for pid, c in json.load(open(f)).items():
                if c in FIVE:
                    pools[c].append({"path": str(T6 / web / "img" / f"{pid}.jpg"), "class_name": c, "source": "internet", "group": web})
    return pools


def features(paths: list[str]) -> np.ndarray:
    sys.path.insert(0, str(REPO / "src"))
    from clip_model import ClipModel
    from train_clip_head import clip_features
    return clip_features(ClipModel("cpu"), [Path(p) for p in paths], cache=CACHE).numpy()


def eval_bank() -> np.ndarray:
    """CLIP features of every evaluation photo (Valid, Test, first internet 221, 160, Asian cucumbers, internet 241, any-photo,
    new-vegetable tests: the T3 bank; T3; T4; weak-spot development share) - the photos an added photo must not copy."""
    t4 = [str(S / "t4/img" / f"{r['id']}.jpg") for r in csv.DictReader(open(REPO / "results/t4_test/t4_test.csv", encoding="utf-8"))]
    dev = [r["path"] for r in csv.DictReader(open(S / "t5/lists/dev.csv", encoding="utf-8"))]
    return np.concatenate([np.load(S / "t3/bank.npy"), np.load(S / "t3/t3_feats.npz")["f"], features(t4 + dev)]).astype(np.float32)


def lists():
    collapsed = {k for k, v in json.load(open(T6 / "collapse_labels.json")).items() if v == "collapsed"}
    base = [r for r in base_rows() if r["path"] not in collapsed]
    pools = fill_pools()
    Fb = features([r["path"] for r in base])
    bank = np.concatenate([Fb, eval_bank()])
    added_f: list[np.ndarray] = []
    rng = random.Random(42)
    final, report, skipped = [], {}, Counter()
    for s in SPECIES:
        side = {c: [r for r in base if r["class_name"] == c] for c in (f"{s}_fresh", f"{s}_rotten")}
        start = {c: len(v) for c, v in side.items()}
        target = max(MIN_T, min(start.values()))
        for c, rows in side.items():
            if len(rows) >= target:
                continue
            pool = pools[c]
            Fp = features([r["path"] for r in pool]) if pool else np.zeros((0, 512), np.float32)
            for r, f in zip(pool, Fp):
                if len(rows) >= target:
                    break
                near = float((bank @ f).max())
                if added_f:
                    near = max(near, float((np.stack(added_f) @ f).max()))
                if near >= COPY:
                    skipped[(c, r["source"])] += 1
                    continue
                rows.append(r); added_f.append(f)
        t = min(target, *(len(v) for v in side.values()))
        for c, rows in side.items():
            if len(rows) > t:
                keep = [r for r in rows if r["group"] == "asian_cucumber"] if c == "cucumber_fresh" else []
                rest = sorted((r for r in rows if r not in keep), key=lambda r: r["path"])
                side[c] = keep + rng.sample(rest, t - len(keep))
        report[s] = {"target": target, "T": t, **{c: {"TV-12_after_collapsed_removed": start[c], "final": len(v),
                                                         "by_source": dict(Counter(r["source"] for r in v))} for c, v in side.items()}}
        final += side[f"{s}_fresh"] + side[f"{s}_rotten"]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "train_five.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "class_name", "source", "group"]); w.writeheader(); w.writerows(final)
    report["collapsed_removed"] = len(collapsed)
    report["copies_skipped"] = {f"{c} {src}": n for (c, src), n in skipped.items()}
    report["asian_fresh_cucumbers_kept"] = sum(r["group"] == "asian_cucumber" for r in final)
    json.dump(report, open(HERE / "b1_counts.json", "w"), indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1, ensure_ascii=False))


def b1():
    """Manifest = Valid (for the temperature) + the Train rows kept in the balanced list; every other photo of the list goes
    in the extra CSV with the 2,850 not-five photos. Paths are absolute, so the feature cache of TV-12's head is reused."""
    import build as t3ws  # experiments/t3_weak_spots/build.py: same head arguments and text-evidence refit
    man = list(csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")))
    five = list(csv.DictReader(open(OUT / "train_five.csv", encoding="utf-8")))
    keep = {r["path"] for r in five if r["source"] == "manifest_train"}
    rows = [r | {"image_path": str(REPO / "dataset" / r["image_path"])} for r in man
            if r["split"] == "val" or (r["split"] == "train" and str(REPO / "dataset" / r["image_path"]) in keep)]
    with open(OUT / "manifest_B1.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(man[0].keys())); w.writeheader(); w.writerows(rows)
    other = [r for r in csv.DictReader(open(S / "negx/extra_train_N3.csv", encoding="utf-8")) if r["class_name"] == "not_five"]
    with open(OUT / "extra_train_B1.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "class_name", "group"]); w.writeheader()
        w.writerows([{"path": r["path"], "class_name": r["class_name"], "group": r["source"]} for r in five if r["source"] != "manifest_train"] + other)
    print("B1 head data:", Counter(r["class_name"] for r in five), "+", len(other), "not_five;",
          sum(r["split"] == "train" for r in rows), "of them from the manifest", flush=True)
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    out = W / "B1.pth"
    shutil.copy(REPO / "model.pth", out)
    t3ws.run("src/train_clip_head.py", "--model", out, "--manifest", OUT / "manifest_B1.csv", "--extra-photos", t3ws.LVIS,
             "--extra-train", OUT / "extra_train_B1.csv", *t3ws.HEAD_ARGS)
    t3ws.refit_text_evidence(out, tv12)
    ck = torch.load(out, map_location="cpu", weights_only=True)
    assert torch.equal(ck["new_vegetable_head"]["coef"], tv12["new_vegetable_head"]["coef"])
    ck["version"] = "candidate B1 (experiments/balanced_clip_head)"
    torch.save(ck, out)
    print("B1 saved", out, "threshold", round(ck["clip_head"]["unknown_threshold"], 4))


if __name__ == "__main__":
    {"lists": lists, "B1": b1}[sys.argv[1]]()
