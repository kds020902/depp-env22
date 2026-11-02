"""Build the candidates of PREREG.md from the reviewed photo lists (working folder t4/lists, record script).

    python experiments/t3_weak_spots/build.py check   # rebuilding TV-12's two CLIP heads from their data must give TV-12 back
    python experiments/t3_weak_spots/build.py K1      # CLIP head + new-vegetable head with the training share added
    python experiments/t3_weak_spots/build.py data    # manifest for the vegetable model: Train + training share, no Test rows
    python experiments/t3_weak_spots/build.py K2      # after training K2_s42 / K2_s43 on that manifest: their best seed + K1's heads

Same rules as TV-12 everywhere: the CLIP head is trained by src/train_clip_head.py with the light watermark copies; the
not-five threshold with text evidence (alpha 0.25) is refitted so that 95% of Valid + 192 real-scene crops pass; tau of the
new-vegetable head stays 0.8 (PREREG addendum); for K2 the vegetable model is calibrated by src/calibrate.py (LVIS crops),
the combined temperature is fitted on Valid and the heat-map scale is the median peak over the Valid rotten photos.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "watermark"))
from common import REPO, S, W, clip_features, dev_sets  # noqa: E402
from model_utils import (  # noqa: E402
    CLASS_NAMES, CONDITION_NAMES, clip_head_logits, clip_unknown_score, combined_logits, fit_temperature,
    forward_with_rot_maps, input_size, load_model, make_transform, open_image,
)

LISTS = S / "t4/lists"
LVIS = S / "negx/lvis_veg"
EXTRA = S / "negx/extra_train_N3.csv"
HEAD_DATA = REPO / "experiments/new_vegetables/v7/head_data.npz"
HEAD_ARGS = ["--clip-check-only", "--watermark-fresh", "1.0", "--watermark-rotten", "0.25", "--watermark-other", "0.5",
             "--watermark-style", "light", "--feature-cache", str(W / "head_features.npz")]
FIVE = CLASS_NAMES[:10]
K2DATA = S / "k2data"


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=REPO)


def train_rows() -> list[dict]:
    return list(csv.DictReader(open(LISTS / "train.csv", encoding="utf-8")))


def write_extra_csv(path: Path, add: bool) -> Path:
    rows = list(csv.DictReader(open(EXTRA, encoding="utf-8")))
    if add:
        rows += [{"path": r["path"], "class_name": "not_five" if r["class_name"] == "rotten_fruit" else r["class_name"],
                  "group": "t3_weak_spots"} for r in train_rows() if r["class_name"] in FIVE or r["class_name"] == "rotten_fruit"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "class_name", "group"]); w.writeheader(); w.writerows(rows)
    return path


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


def new_veg_head(path: Path, add: bool, data_out: Path):
    data = dict(np.load(HEAD_DATA))
    if add:
        from clip_model import ClipModel
        clip = ClipModel("cpu")
        rows = [r for r in train_rows() if r["class_name"] in CLASS_NAMES[10:]]
        with torch.inference_mode():
            X = torch.cat([clip.image_features([open_image(r["path"]) for r in rows[i:i + 32]]).cpu() for i in range(0, len(rows), 32)]).numpy()
        data["train_X"] = np.concatenate([data["train_X"], X.astype(np.float32)])
        data["train_y"] = np.concatenate([data["train_y"], np.array([CLASS_NAMES.index(r["class_name"]) for r in rows], np.int16)])
        data["train_tag"] = np.concatenate([data["train_tag"], np.array(["t3_weak_spots"] * len(rows))])
        print("new-vegetable head: added", {c: sum(r["class_name"] == c for r in rows) for c in CLASS_NAMES[10:]}, flush=True)
    np.savez(data_out, **data)
    run("src/train_new_veg_head.py", "--model", path, "--data", data_out)
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck["new_vegetable_head"]["switch_probability"] = 0.8
    ck["new_vegetable_head"]["switch_rule"] = "tau 0.8 as in TV-12 (ZS8), experiments/t3_weak_spots/PREREG.md"
    torch.save(ck, path)


def heads(tag: str, add: bool) -> Path:
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    out = W / f"{tag}.pth"
    shutil.copy(REPO / "model.pth", out)
    run("src/train_clip_head.py", "--model", out, "--extra-photos", LVIS, "--extra-train", write_extra_csv(W / f"extra_train_{tag}.csv", add), *HEAD_ARGS)
    refit_text_evidence(out, tv12)
    new_veg_head(out, add, W / f"head_data_{tag}.npz")
    ck = torch.load(out, map_location="cpu", weights_only=True)
    ck["version"] = f"candidate {tag} (experiments/t3_weak_spots)"
    torch.save(ck, out)
    return out


def check():
    """TV-12 rebuilt from its own data (no additions) must match model.pth closely (pipeline check)."""
    out = heads("K0", add=False)
    a = torch.load(out, map_location="cpu", weights_only=True); b = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    sets = dev_sets(); F = clip_features(sets)
    from common import system
    B = dict(np.load(W / "b_WC.npz"))
    same = {n: int((system(a, B[n], F[n])["index"] == system(b, B[n], F[n])["index"]).sum()) for n in ("valid", "odev", "o160", "lvis", "newval")}
    diff = {"clip_threshold": (a["clip_head"]["unknown_threshold"], b["clip_head"]["unknown_threshold"]),
            "combined_T": (a["clip_head"]["temperature"], b["clip_head"]["temperature"]),
            "new_veg_threshold": (a["new_vegetable_head"]["unknown_threshold"], b["new_vegetable_head"]["unknown_threshold"]),
            "same_answers": same}
    print(json.dumps(diff, indent=1))


def k2_manifest():
    """Train + Valid rows of dataset/manifest.csv (no Test rows, so training never scores Test) + the training share."""
    os.makedirs(K2DATA / "add", exist_ok=True)
    for d in ("train", "valid"):
        if not (K2DATA / d).exists():
            os.symlink(REPO / "dataset" / d, K2DATA / d)
    base = [r for r in csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")) if r["split"] in ("train", "val")]
    fields = list(base[0].keys())
    add = []
    for r in train_rows():
        if r["class_name"] not in FIVE:
            continue
        dst = K2DATA / "add" / f"{r['id']}.jpg"
        if not dst.exists():
            shutil.copy(r["path"], dst)
        sha = hashlib.sha256(dst.read_bytes()).hexdigest()
        row = {k: "" for k in fields}
        row.update(image_id=f"t3ws_{r['id']}", image_path=f"add/{r['id']}.jpg", class_name=r["class_name"],
                   class_id=str(CLASS_NAMES.index(r["class_name"])), species=r["class_name"].split("_")[0],
                   condition=r["class_name"].split("_")[1], split="train", group_id=f"t3ws_{r['user']}",
                   session_id=f"t3ws_{r['user']}", source_dataset=r.get("source", ""), source_url=r.get("page", ""),
                   label_origin="visual_review (experiments/t3_weak_spots)", sha256=sha)
        add.append(row)
    with open(K2DATA / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(base + add)
    from collections import Counter
    print("K2 manifest:", len(base), "+", len(add), Counter(r["class_name"] for r in add))


def k2():
    loss = {s: json.load(open(W / f"K2_s{s}/metrics.json"))["metrics"]["val"]["loss"] for s in (42, 43)}
    run_dir = W / f"K2_s{min(loss, key=loss.get)}"
    print("Valid loss per seed", loss, "->", run_dir.name, flush=True)
    out = W / "K2.pth"
    shutil.copy(run_dir / "model.pth", out)
    run("src/calibrate.py", "--model", out, "--extra-photos", LVIS, "--extra-name", "LVIS real-scene vegetable crops")
    ck = torch.load(out, map_location="cpu", weights_only=True)
    k1 = torch.load(W / "K1.pth", map_location="cpu", weights_only=True)
    ck["clip_head"] = copy.deepcopy(k1["clip_head"]); ck["new_vegetable_head"] = copy.deepcopy(k1["new_vegetable_head"])
    sets = dev_sets(); F = clip_features(sets); paths, y = sets["valid"]
    model, _ = load_model(out, torch.device("cpu")); tf = make_transform(input_size(ck))
    with torch.inference_mode():
        b = torch.cat([model(torch.stack([tf(open_image(p)) for p in paths[i:i + 32]])) for i in range(0, len(paths), 32)])
    ck["clip_head"]["temperature"] = fit_temperature(combined_logits(b, clip_head_logits(torch.tensor(F["valid"]), ck), ck), torch.tensor(y))
    rotten = [p for p, c in zip(paths, y) if c % len(CONDITION_NAMES) == 1]; peaks = []
    with torch.inference_mode():
        for i in range(0, len(rotten), 32):
            logits, maps = forward_with_rot_maps(model, ck, torch.stack([tf(open_image(p)) for p in rotten[i:i + 32]]))
            species = logits.argmax(1) // len(CONDITION_NAMES)
            peaks += [float(maps[k, s].max()) for k, s in enumerate(species)]
    ck["rot_map"] = {**k1["rot_map"], "scale": float(np.median(peaks))}
    ck["version"] = "candidate K2 (experiments/t3_weak_spots)"
    ck.setdefault("watermark_round", k1.get("watermark_round"))
    torch.save(ck, out)
    print("K2 combined T", round(ck["clip_head"]["temperature"], 4), "rot scale", round(ck["rot_map"]["scale"], 3), flush=True)


if __name__ == "__main__":
    {"check": check, "K1": lambda: heads("K1", add=True), "data": k2_manifest, "K2": k2}[sys.argv[1]]()
