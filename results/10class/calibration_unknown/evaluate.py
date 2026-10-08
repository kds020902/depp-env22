"""How the calibrated probability and the "학습된 데이터가 아닙니다" check in model.pth were chosen and scored.

    # 1. outside photos (fruits from the same Kaggle archive as the vegetables, non-food COCO val2017 photos)
    python results/10class/calibration_unknown/evaluate.py fetch --kaggle-zip fruits-and-vegetables-dataset.zip --photos outside_photos
    # 2. scores -> results.json (needs scikit-learn)
    python results/10class/calibration_unknown/evaluate.py run --photos outside_photos

Temperature and threshold are fitted on Valid only. The method is chosen on Valid against the dev halves of the
outside sets; the test halves and the Test split are scored once, after the choice.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
from model_utils import load_model, make_transform, open_image  # noqa: E402

BIN_EDGES = [0, 0.5, 0.7, 0.9, 0.97, 1.0]


def outside_rows():
    with (HERE / "outside_photos.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fetch(args):
    archive = zipfile.ZipFile(args.kaggle_zip)
    for row in outside_rows():
        out = args.photos / row["set"] / row["file"]
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if row["set"] == "fruits":
            out.write_bytes(archive.read(row["source"].split(" :: ", 1)[1]))
        else:
            with urllib.request.urlopen(row["source"], timeout=60) as response:
                out.write_bytes(response.read())


def features(model, transform, paths):
    logits, feats = [], []
    with torch.inference_mode():
        for i in range(0, len(paths), 32):
            x = torch.stack([transform(open_image(p)) for p in paths[i:i + 32]])
            f = torch.flatten(model.avgpool(model.features(x)), 1)
            logits.append(model.classifier(f).numpy())
            feats.append(f.numpy())
    return np.concatenate(logits), np.concatenate(feats)


def softmax(z):
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def ece(p, y, bins=15):
    conf, pred, edges, total = p.max(1), p.argmax(1), np.linspace(0, 1, bins + 1), 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf > a) & (conf <= b)
        if m.any():
            total += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return round(float(total), 4)


def nll(p, y):
    return round(float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()), 4)


def run(args):
    from sklearn.covariance import LedoitWolf
    from sklearn.metrics import roc_auc_score

    torch.set_num_threads(args.threads)
    model, checkpoint = load_model(REPO / "model.pth", torch.device("cpu"))
    transform = make_transform(224)
    with (REPO / "dataset" / "manifest.csv").open(encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    data = {}
    for split in ("train", "val", "test"):
        rows = [r for r in manifest if r["split"] == split]
        data[split] = features(model, transform, [REPO / "dataset" / r["image_path"] for r in rows])
        data[split + "_y"] = np.array([int(r["class_id"]) for r in rows])
    outside = outside_rows()
    halves = {}
    for name in ("fruits", "coco"):
        rows = [r for r in outside if r["set"] == name]
        data[name] = features(model, transform, [args.photos / name / r["file"] for r in rows])
        halves[name] = {h: np.array([r["half"] == h for r in rows]) for h in ("dev", "test")}

    # calibration: one temperature fitted on Valid (same code as model.pth, see model_utils.fit_temperature)
    temperature = checkpoint["calibration"]["temperature"]
    result = {"temperature": round(temperature, 4)}
    for split in ("val", "test"):
        y = data[split + "_y"]
        p0, p1 = softmax(data[split][0]), softmax(data[split][0] / temperature)
        conf, right = p1.max(1), p1.argmax(1) == y
        bins = []
        for a, b in zip(BIN_EDGES[:-1], BIN_EDGES[1:]):
            m = (conf >= a) & ((conf < b) if b < 1 else (conf <= b))
            bins.append({"probability": f"{a:.0%}-{b:.0%}", "photos": int(m.sum()),
                         "accuracy": round(float(right[m].mean()), 3) if m.any() else None})
        result[split] = {"accuracy": round(float(right.mean()), 4),
                         "ece_before": ece(p0, y), "ece_after": ece(p1, y), "nll_before": nll(p0, y),
                         "nll_after": nll(p1, y), "mean_confidence_before": round(float(p0.max(1).mean()), 4),
                         "mean_confidence_after": round(float(conf.mean()), 4), "reliability_after": bins}

    # unknown-photo scores, higher = less like any trained class
    def l2n(x):
        return x / np.linalg.norm(x, axis=1, keepdims=True)
    train_feats, train_y = data["train"][1], data["train_y"]
    train_unit = l2n(train_feats)
    means = np.stack([train_feats[train_y == c].mean(0) for c in range(len(checkpoint["class_names"]))])
    precision = LedoitWolf().fit(train_feats - means[train_y]).precision_

    def score(method, name):
        logits, feats = data[name]
        if method == "msp":
            return -softmax(logits / temperature).max(1)
        if method == "energy":
            return -np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) - logits.max(1)
        if method.startswith("knn"):
            return 1 - np.sort(l2n(feats) @ train_unit.T, axis=1)[:, -int(method[3:])]
        return np.stack([np.einsum("ij,jk,ik->i", feats - m, precision, feats - m) for m in means], 1).min(1)

    dev_auroc = {}
    for method in ("msp", "energy", "knn1", "knn5", "knn10", "maha"):
        inside = score(method, "val")
        out = np.concatenate([score(method, n)[halves[n]["dev"]] for n in ("fruits", "coco")])
        labels = np.r_[np.zeros(len(inside)), np.ones(len(out))]
        dev_auroc[method] = round(float(roc_auc_score(labels, np.r_[inside, out])), 4)
    chosen = max(dev_auroc, key=dev_auroc.get)
    threshold = float(np.quantile(score(chosen, "val"), 0.95))
    result.update({"dev_auroc": dev_auroc, "chosen": chosen, "threshold": threshold,
                   "threshold_in_model_pth": checkpoint["unknown_check"]["threshold"]})
    for name in ("fruits", "coco"):
        rejected = score(chosen, name) > threshold
        for h in ("dev", "test"):
            result[f"{name}_{h}_rejected"] = f"{int(rejected[halves[name][h]].sum())}/{int(halves[name][h].sum())}"
    fruit_rows = [r for r in outside if r["set"] == "fruits"]
    rejected, test_half = score(chosen, "fruits") > threshold, halves["fruits"]["test"]
    result["fruits_test_passed_by_kind"] = {
        kind: f"{sum(1 for r, x, t in zip(fruit_rows, rejected, test_half) if t and not x and r['label'] == kind)}"
              f"/{sum(1 for r, t in zip(fruit_rows, test_half) if t and r['label'] == kind)}"
        for kind in sorted({r["label"] for r in fruit_rows})}
    rejected = score(chosen, "test") > threshold
    right = data["test"][0].argmax(1) == data["test_y"]
    names = checkpoint["class_names"]
    result.update({
        "test_rejected": f"{int(rejected.sum())}/{len(rejected)}",
        "test_rejected_right": int((rejected & right).sum()), "test_rejected_wrong": int((rejected & ~right).sum()),
        "test_rejected_by_class": {names[c]: int(rejected[data["test_y"] == c].sum())
                                   for c in range(len(names)) if rejected[data["test_y"] == c].any()},
        "test_accuracy_all": round(float(right.mean()), 4),
        "test_accuracy_accepted": round(float(right[~rejected].mean()), 4),
    })
    (HERE / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--kaggle-zip", type=Path, required=True)
    f.add_argument("--photos", type=Path, default=Path("outside_photos"))
    r = sub.add_parser("run")
    r.add_argument("--photos", type=Path, default=Path("outside_photos"))
    r.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    fetch(args) if args.command == "fetch" else run(args)


if __name__ == "__main__":
    main()
