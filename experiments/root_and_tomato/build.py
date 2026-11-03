"""experiments/root_and_tomato (PREREG.md): build B0' (TV-12's recipe on the recovered training photos) and R1 (+ new photos).

    python experiments/root_and_tomato/build.py dedup        # CLIP cosine >= 0.95 to test / development / dataset photos -> left out
    python experiments/root_and_tomato/build.py B0p          # TV-12's two heads rebuilt, no new photos
    python experiments/root_and_tomato/build.py R1           # + the new photos of new_train.csv that pass dedup
    python experiments/root_and_tomato/build.py R2           # the same with the R2 photos; the new-vegetable head gets every new photo

Same rules as TV-12 (experiments/t3_weak_spots/build.py):
- CLIP head: src/train_clip_head.py with the light watermark copies (fresh 1.0 / rotten 0.25 / not-five 0.5), C=10.
- TV-12's text evidence (alpha 0.25) put back, and the not-five threshold refitted so that 95% of Valid + the 192 LVIS crops pass.
- New-vegetable head: src/train_new_veg_head.py on head_data.npz (+ new onion / sweet potato / green onion rows), tau 0.8.
Photos are in the working folder ($DEPP_WORK): recovered head photos (tools/rebuild_head_photos.py) and rt/img (photos.py).
"""

from __future__ import annotations

import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
from decision import clip_head_logits, clip_unknown_score  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402

S = Path(os.environ["DEPP_WORK"])
W = S / "wm"
HEAD_DATA = REPO / "experiments/new_vegetables/v7/head_data.npz"
LVIS = S / "negx/lvis_veg"
RT = S / "rt/img"
HEAD_ARGS = ["--clip-check-only", "--watermark-fresh", "1.0", "--watermark-rotten", "0.25", "--watermark-other", "0.5",
             "--watermark-style", "light", "--feature-cache", str(W / "head_features.npz")]
DEDUP = HERE / "dedup.json"


def run(*args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=REPO, env={**os.environ, "DEPP_WORK": str(S)})


def new_rows(tag: str) -> list[dict]:
    """The photos of new_train.csv that passed dedup; R1 leaves out the rows added for R2 (note starting with "R2")."""
    rows = [r for r in csv.DictReader(open(HERE / "new_train.csv", encoding="utf-8")) if tag != "R1" or not r["note"].startswith("R2")]
    out = set(json.load(open(DEDUP))["left_out"]) if DEDUP.exists() else None
    assert out is not None, "run dedup first"
    return [r for r in rows if r["id"] not in out]


def features(paths: list[str]) -> np.ndarray:
    from clip_model import ClipModel
    from images import open_image
    clip = ClipModel("cpu")
    with torch.inference_mode():
        return torch.cat([clip.image_features([open_image(p) for p in paths[i:i + 32]]).cpu() for i in range(0, len(paths), 32)]).numpy()


def dedup():
    """New photos with CLIP cosine >= 0.95 to any T3-T6 photo, any dataset/ photo, any stored development photo, or to an
    earlier new photo are left out. Image-to-image similarity only (no model answer is looked at)."""
    rows = list(csv.DictReader(open(HERE / "new_train.csv", encoding="utf-8")))
    X = features([str(RT / f"{r['id']}.jpg") for r in rows])
    bank, names = [], []
    cache = W / "dedup_bank.npz"
    if cache.exists():
        z = np.load(cache); bank, names = [z["f"]], z["names"].tolist()
    else:
        paths = []
        for t, folder in (("t3", "t3/img"), ("t4", "t4/img"), ("t5", "t5test/img"), ("t6", "t6test/img")):
            for r in csv.DictReader(open(REPO / f"results/{t}_test/{t}_test.csv", encoding="utf-8")):
                if (S / folder / f"{r['id']}.jpg").exists():
                    paths.append(str(S / folder / f"{r['id']}.jpg")); names.append(f"{t}:{r['id']}")
        for r in csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")):
            paths.append(str(REPO / "dataset" / r["image_path"])); names.append(f"dataset:{r['image_path']}")
        bank = [features(paths)]
        stored = np.load(HEAD_DATA)
        for n in ("valid", "odev", "o160", "asian37", "fruit374", "coco400", "lvis", "newval"):
            bank.append(stored[f"dev_{n}_X"]); names += [f"{n}#{i}" for i in range(len(stored[f"dev_{n}_X"]))]
        np.savez(cache, f=np.concatenate(bank), names=np.array(names))
        bank = [np.concatenate(bank)]
    B = torch.nn.functional.normalize(torch.tensor(bank[0]).float(), dim=1); Xn = torch.nn.functional.normalize(torch.tensor(X).float(), dim=1)
    sim = Xn @ B.T; best, idx = sim.max(1)
    left_out, why = [], {}
    for i, r in enumerate(rows):
        if float(best[i]) >= 0.95:
            left_out.append(r["id"]); why[r["id"]] = [names[int(idx[i])], round(float(best[i]), 4)]
    inner = Xn @ Xn.T
    for i, r in enumerate(rows):
        if r["id"] in left_out:
            continue
        for j in range(i):
            if rows[j]["id"] not in left_out and float(inner[i, j]) >= 0.95:
                left_out.append(r["id"]); why[r["id"]] = [rows[j]["id"], round(float(inner[i, j]), 4)]; break
    counted = {"t6": [n for n in names if n.startswith("t6:")].__len__(), "bank": len(names)}
    json.dump({"left_out": left_out, "why": why, "checked_against": counted, "n_new": len(rows),
               "max_cos_to_bank_kept": round(float(max(best[i] for i, r in enumerate(rows) if r["id"] not in left_out)), 4)},
              open(DEDUP, "w"), indent=1)
    print("left out", len(left_out), "of", len(rows), why)


def refit_text_evidence(path: Path, tv12: dict):
    """TV-12's text evidence into the new CLIP head; not-five threshold so that 95% of Valid + LVIS crops pass (stored features)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    ck["clip_head"]["text_evidence"] = copy.deepcopy(tv12["clip_head"]["text_evidence"])
    stored = np.load(HEAD_DATA)
    F = [torch.tensor(stored[f"dev_{n}_X"]) for n in ("valid", "lvis")]
    scores = [clip_unknown_score(clip_head_logits(f, ck), ck, f) for f in F]
    ck["clip_head"]["tv11_unknown_threshold"] = ck["clip_head"]["unknown_threshold"]
    ck["clip_head"]["unknown_threshold"] = float(torch.quantile(torch.cat(scores), 0.95))
    torch.save(ck, path)


def new_veg_head(path: Path, rows: list[dict], data_out: Path, all_classes: bool = False):
    """R1: only the onion / sweet potato / green onion photos are added; R2 (PREREG addendum 1): every new photo, since this
    head also learns the five vegetables' ten classes."""
    data = dict(np.load(HEAD_DATA))
    add = [r for r in rows if all_classes or r["class_name"] in CLASS_NAMES[10:]]
    if add:
        X = features([str(RT / f"{r['id']}.jpg") for r in add])
        data["train_X"] = np.concatenate([data["train_X"], X.astype(np.float32)])
        data["train_y"] = np.concatenate([data["train_y"], np.array([CLASS_NAMES.index(r["class_name"]) for r in add], np.int16)])
        data["train_tag"] = np.concatenate([data["train_tag"], np.array(["root_and_tomato"] * len(add))])
    print("new-vegetable head: added", {c: n for c in CLASS_NAMES if (n := sum(r["class_name"] == c for r in add))}, flush=True)
    np.savez(data_out, **data)
    run("src/train_new_veg_head.py", "--model", path, "--data", data_out, "--tau", "0.8")


def heads(tag: str, add: bool, skip_clip_head: bool = False) -> Path:
    tv12 = torch.load(REPO / "model.pth", map_location="cpu", weights_only=True)
    out = W / f"{tag}.pth"
    rows = new_rows(tag) if add else []
    extra = W / f"extra_train_{tag}.csv"
    base = list(csv.DictReader(open(W / "extra_train_B0p.csv", encoding="utf-8")))   # the recovered TV-12 head photos
    with open(extra, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "class_name", "group"]); w.writeheader(); w.writerows(base)
        w.writerows({"path": str(RT / f"{r['id']}.jpg"), "class_name": r["class_name"], "group": "root_and_tomato"}
                    for r in rows if r["class_name"] in CLASS_NAMES[:10])
    if not skip_clip_head:
        shutil.copy(REPO / "model.pth", out)
        run("src/train_clip_head.py", "--model", out, "--extra-photos", LVIS, "--extra-train", extra, "--threads", "4", *HEAD_ARGS)
    refit_text_evidence(out, tv12)
    new_veg_head(out, rows, W / f"head_data_{tag}.npz", all_classes=tag != "R1")
    ck = torch.load(out, map_location="cpu", weights_only=True)
    ck["version"] = f"{tag} (experiments/root_and_tomato)"
    ck["new_vegetable_head"]["switch_rule"] = "tau 0.8 as in TV-12 (ZS8), experiments/root_and_tomato/PREREG.md"
    torch.save(ck, out)
    print("saved", out, {"five-vegetable rows added": sum(r["class_name"] in CLASS_NAMES[:10] for r in rows)}, flush=True)
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dedup":
        dedup()
    elif cmd == "B0p":
        heads("B0p", add=False, skip_clip_head="--skip-clip-head" in sys.argv)
    else:
        heads(cmd, add=True)
