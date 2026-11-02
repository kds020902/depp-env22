"""Shared parts of the watermark round: development photo sets, cached model outputs, the full 8-vegetable decision.

The photo lists point into the working folder of this project (photos are not in the repository for licence reasons),
so these scripts are a record of what was run.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from decision import clip_class_logits, clip_head_logits, clip_unknown_score, decide  # noqa: E402
from model_utils import CLASS_NAMES, input_size, load_model, make_transform, open_image  # noqa: E402
from stamps import STAMPS  # noqa: E402

S = Path(os.environ["DEPP_WORK"])
W = S / "wm"
TAU_GRID = (0.5, 0.7, 0.8, 0.9, 0.95)
OLD_DEV = ("valid", "odev", "o160", "asian37", "lvis")


def dev_sets() -> dict[str, tuple[list[str], np.ndarray | None]]:
    """name -> (photo paths, class indices or None for photos that should be turned away / have no label)."""
    sets = json.load(open(S / "clip/sets.json"))
    out = {}
    for name, key in (("valid", "valid"), ("odev", "odev"), ("o160", "o160"), ("asian37", "asian37")):
        out[name] = ([p for p, _ in sets[key]], np.array([CLASS_NAMES.index(c) for _, c in sets[key]]))
    for name, key in (("fruit374", "fruit374"), ("coco400", "coco400"), ("lvis", "lvis_veg")):
        out[name] = ([p for p, _ in sets[key]], None)
    pool = [r for r in json.load(open(S / "v7/pool.json")) if r["split"] == "val"]
    out["newval"] = ([r["path"] for r in pool], np.array([CLASS_NAMES.index(r["class_name"]) for r in pool]))
    return out


def valid_watermarks() -> np.ndarray:
    """1 where the Valid photo (in dev_sets()['valid'] order) carries a real watermark."""
    rows = {r["image_path"]: int(r["watermark"]) for r in csv.DictReader(open(REPO / "experiments/watermark/valid_test_watermarks.csv"))}
    paths, _ = dev_sets()["valid"]
    return np.array([rows[str(Path(p).relative_to(REPO / "dataset"))] for p in paths])


def _images(paths: list[str], stamp: str | None):
    for p in paths:
        image = open_image(p)
        yield STAMPS[stamp](image) if stamp else image


def clip_features(sets, cache: Path = W / "clip_cache.npz") -> dict[str, np.ndarray]:
    """CLIP image features (the same model as the CLIP heads); they do not depend on the candidate, so computed once."""
    from clip_model import ClipModel
    have = dict(np.load(cache)) if cache.exists() else {}
    clip = None
    for name, stamp in [(n, None) for n in sets] + [("valid_S1", "S1"), ("valid_S2", "S2")]:
        if name in have:
            continue
        clip = clip or ClipModel("cpu")
        paths = sets["valid" if stamp else name][0]
        images = list(_images(paths, stamp))
        with torch.inference_mode():
            have[name] = torch.cat([clip.image_features(images[i:i + 32]).cpu() for i in range(0, len(images), 32)]).numpy()
        np.savez(cache, **have)
    return have


def vegetable_logits(model_path: Path, tag: str, sets) -> dict[str, np.ndarray]:
    cache = W / f"b_{tag}.npz"
    have = dict(np.load(cache)) if cache.exists() else {}
    model, checkpoint = load_model(model_path, torch.device("cpu")); transform = make_transform(input_size(checkpoint))
    for name, stamp in [(n, None) for n in sets] + [("valid_S1", "S1"), ("valid_S2", "S2")]:
        if name in have:
            continue
        paths = sets["valid" if stamp else name][0]
        out = []
        with torch.inference_mode():
            batch = []
            for image in _images(paths, stamp):
                batch.append(transform(image))
                if len(batch) == 32:
                    out.append(model(torch.stack(batch))); batch = []
            if batch:
                out.append(model(torch.stack(batch)))
        have[name] = torch.cat(out).numpy()
        np.savez(cache, **have)
    return have


def system(checkpoint: dict, b: np.ndarray, f: np.ndarray):
    """Five-vegetable probabilities and unknown flags, then the full 8-vegetable answer (chosen class, turned away)."""
    logits, feats = torch.tensor(b), torch.tensor(f)
    a = decide(logits, checkpoint, feats)
    return {"five": a.five.numpy(), "five_unknown": a.five_unknown.numpy(), "index": a.index.numpy(),
            "unknown": a.unknown.numpy(), "switched": a.switched.numpy(),
            "vegetable_model": (logits / checkpoint["calibration"]["temperature"]).softmax(1).numpy(),
            "clip_head": clip_class_logits(clip_head_logits(feats, checkpoint), checkpoint).softmax(1).numpy()}


def kept_right_per_class(checkpoint: dict, names=("valid", "odev", "o160", "asian37")) -> dict:
    """set -> class -> "kept and right / photos" on the labelled development sets (TV-12's vegetable-model outputs, b_WC)."""
    sets = dev_sets(); F = clip_features(sets); B = dict(np.load(W / "b_WC.npz")); out = {}
    for name in names:
        r = system(checkpoint, B[name], F[name]); y = sets[name][1]
        out[name] = {CLASS_NAMES[c]: f"{int(((r['index'] == c) & ~r['unknown'])[y == c].sum())}/{int((y == c).sum())}"
                     for c in range(10) if (y == c).any()}
    return out


def choose_tau(checkpoint: dict, B: dict, F: dict) -> tuple[float, dict]:
    """The V5/V6/V7 rule: the smallest tau that moves at most one old-vegetable development photo, which the
    five-vegetable system answered, to a new vegetable."""
    head = checkpoint["new_vegetable_head"]; keep = head["switch_probability"]; grid = {}
    for tau in TAU_GRID:
        head["switch_probability"] = tau
        grid[tau] = sum(int((~r["five_unknown"] & r["switched"]).sum()) for r in (system(checkpoint, B[n], F[n]) for n in OLD_DEV))
    head["switch_probability"] = keep
    return min([t for t, m in grid.items() if m <= 1] or [TAU_GRID[-1]]), grid

# ---------- photo sets of the last scorings (experiments/watermark/final.py) ----------
NEW = ("onion", "sweetpotato", "greenonion")


def photo_sets():
    test = [r for r in csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")) if r["split"] == "test"]
    sets = {"test_300": ([str(REPO / "dataset" / r["image_path"]) for r in test], [r["class_name"] for r in test])}
    t2 = list(csv.DictReader(open(REPO / "results/10class/outside_test2/outside_test2.csv", encoding="utf-8")))
    sets["new_test_241"] = ([str(S / f"weak/t2img/{r['id']}.jpg") for r in t2], [r["class_name"] for r in t2])
    rows = list(csv.DictReader(open(S / "arb/anyphoto_test_v2.csv", encoding="utf-8"))); files = json.load(open(S / "arb/local_files.json"))
    sets["any_photo_v2"] = ([files[r["id"]] for r in rows], rows)
    pool = json.load(open(S / "v7/pool.json"))
    for k, rnd in enumerate(("v4", "v5", "v6", "v7"), 1):
        te = [r for r in pool if r["round"] == rnd and r["split"] == "test" and r["class_name"].startswith(NEW)]
        sets[f"new_veg_test{k}"] = ([r["path"] for r in te], [r["class_name"] for r in te])
    return sets


# ---------- rotten fruit photos the CLIP head should turn away (experiments/zero_shot_reject) ----------
def rotten_fruit_features() -> dict[str, np.ndarray]:
    pool = list(csv.DictReader(open(S / "arb/pool/pool.csv", encoding="utf-8"))); pf = np.load(S / "arb/pool/feats.npz")["f"]
    keep = set(np.load(S / "arb/pool/keep_idx.npy").tolist())
    inf = np.load(S / "arb/inat/features.npz"); idx = {str(n): i for i, n in enumerate(inf["ids"])}
    return {"kaggle_fruit": pf[[i for i, r in enumerate(pool) if i in keep and r["group"] == "kaggle_fruit"]],
            "inat_rotten_fruit": inf["f"][[idx[i] for i in json.load(open(S / "arb/pool/inat_train_keep.json"))]],
            "openverse_rotten_fruit": np.load(S / "arb/ov_train_fruit.npz")["f"]}


# ---------- development check: the 8 rules of experiments/t3_weak_spots/PREREG.md ----------
DEV_LISTS = S / "t5/lists"   # weak-spot development share, round 2 (round 1: t4/lists)
WEAK = ("potato_fresh", "carrot_fresh", "tomato_rotten", "sweetpotato_fresh", "rotten_fruit")
PAIR = ("potato_rotten", "tomato_fresh", "carrot_rotten")


def dev_new(lists: Path, tag: str, path: Path):
    """CLIP features (cached once) and vegetable-model logits (cached per candidate) of the development share."""
    rows = list(csv.DictReader(open(lists / "dev.csv", encoding="utf-8")))
    sfx = "" if lists.parent.name == "t4" else "_" + lists.parent.name
    fcache, bcache = W / f"t3ws_dev_clip{sfx}.npz", W / f"t3ws_dev_b_{tag}{sfx}.npz"
    paths = [r["path"] for r in rows]
    if fcache.exists():
        F = np.load(fcache)["f"]
    else:
        from clip_model import ClipModel
        clip = ClipModel("cpu")
        with torch.inference_mode():
            F = torch.cat([clip.image_features([open_image(p) for p in paths[i:i + 32]]).cpu() for i in range(0, len(paths), 32)]).numpy()
        np.savez(fcache, f=F)
    if bcache.exists():
        B = np.load(bcache)["b"]
    else:
        model, ck = load_model(path, torch.device("cpu")); tf = make_transform(input_size(ck))
        with torch.inference_mode():
            B = torch.cat([model(torch.stack([tf(open_image(p)) for p in paths[i:i + 32]])) for i in range(0, len(paths), 32)]).numpy()
        np.savez(bcache, b=B)
    return rows, F, B


def dev_score(path: Path, b_tag: str = "WC", lists: Path = DEV_LISTS) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sets = dev_sets(); F = clip_features(sets)
    B = vegetable_logits(path, b_tag, sets) if b_tag != "WC" else dict(np.load(W / "b_WC.npz"))
    out = {}
    for name in ("valid", "odev", "o160", "asian37", "coco400", "lvis", "newval"):
        r = system(ck, B[name], F[name]); y = sets[name][1]
        if y is None:
            out[name] = {"turned_away": int(r["unknown"].sum()), "to_new_vegetable": int(((r["index"] >= 10) & ~r["unknown"]).sum())}
        elif name == "newval":
            out[name] = {"kind_right_kept": int(((r["index"] // 2 == y // 2) & ~r["unknown"]).sum())}
        else:
            out[name] = {"kept_correct": int(((r["index"] == y) & ~r["unknown"]).sum()),
                         "to_new_vegetable": int(((r["index"] >= 10) & ~r["unknown"]).sum())}
    out["old_dev_to_new"] = sum(out[n]["to_new_vegetable"] for n in OLD_DEV)
    rf = rotten_fruit_features(); X = np.concatenate(list(rf.values()))
    out["old_rotten_fruit_761_turned_away"] = int(system(ck, np.zeros((len(X), 10), np.float32), X)["unknown"].sum())
    rows, Fn, Bn = dev_new(lists, b_tag, path)
    r = system(ck, Bn, Fn); per = {}
    for c in sorted({x["class_name"] for x in rows}):
        m = np.array([x["class_name"] == c for x in rows])
        if c == "rotten_fruit":
            good = r["unknown"][m]
        elif c.startswith("sweetpotato"):
            good = (r["index"][m] // 2 == CLASS_NAMES.index(c) // 2) & ~r["unknown"][m]
        else:
            good = (r["index"][m] == CLASS_NAMES.index(c)) & ~r["unknown"][m]
        per[c] = [int(good.sum()), int(m.sum())]
    out["new_dev"] = per
    out["weak_total"] = sum(per.get(c, [0])[0] for c in WEAK)
    out["pair_total"] = sum(per.get(c, [0])[0] for c in PAIR)
    return out


def dev_rules(c: dict, base: dict) -> dict:
    """The 8 development rules (PREREG.md): what a candidate (score() output) must keep compared with the base model."""
    d = lambda n, f: c[n][f] - base[n][f]
    return {"Valid kept >= -3": d("valid", "kept_correct") >= -3, "internet 221 kept >= -3": d("odev", "kept_correct") >= -3,
            "160 kept >= -3": d("o160", "kept_correct") >= -3, "Asian 37 kept >= -2": d("asian37", "kept_correct") >= -2,
            "everyday 400 turned away >= -5": d("coco400", "turned_away") >= -5,
            "new-vegetable Valid >= -2": d("newval", "kind_right_kept") >= -2,
            "old-vegetable dev to new <= +1": c["old_dev_to_new"] - base["old_dev_to_new"] <= 1,
            "pair classes >= -2": c["pair_total"] - base["pair_total"] >= -2}


# ---------- a CLIP head trained with TV-12's rules (experiments/t3_weak_spots/build.py) ----------
LVIS = S / "negx/lvis_veg"
HEAD_ARGS = ["--clip-check-only", "--watermark-fresh", "1.0", "--watermark-rotten", "0.25", "--watermark-other", "0.5",
             "--watermark-style", "light", "--feature-cache", str(W / "head_features.npz")]


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=REPO)


def refit_text_evidence(path: Path, tv12: dict):
    """Put TV-12's text evidence into the new CLIP head and refit the not-five threshold by the zero_shot_reject rule."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck["clip_head"]["text_evidence"] = copy.deepcopy(tv12["clip_head"]["text_evidence"])
    F = clip_features(dev_sets())
    scores = [clip_unknown_score(clip_head_logits(torch.tensor(F[n]), ck), ck, torch.tensor(F[n])) for n in ("valid", "lvis")]
    ck["clip_head"]["tv11_unknown_threshold"] = ck["clip_head"]["unknown_threshold"]
    ck["clip_head"]["unknown_threshold"] = float(torch.quantile(torch.cat(scores), 0.95))
    torch.save(ck, path)
    return ck


if __name__ == "__main__":
    os.makedirs(W, exist_ok=True)
    sets = dev_sets()
    print({k: len(v[0]) for k, v in sets.items()})
    F = clip_features(sets)
    print({k: v.shape for k, v in F.items()})
