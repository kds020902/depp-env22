"""Train a six-class vegetable freshness model and evaluate it once on the test split.

Two modes:
  linear   - frozen ImageNet backbone, only the new classifier is trained (the 1st-round method)
  finetune - the whole network is trained with augmentation (backbone at a lower learning rate)
Model selection (early stopping) uses the validation split only; the test split is scored
once with the selected weights at the very end.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from model_utils import (
    CLASS_NAMES, CONDITION_NAMES, SPECIES_NAMES,
    classifier_layer, create_model, make_train_transform, make_transform,
    marginal_probabilities, open_image, preprocess_for, resolve_device, set_classifier_layer, set_seed,
    state_checksum,
)

SPLITS = ("train", "val", "test")


def read_manifest(path: Path) -> dict[str, list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "image_path", "class_name", "class_id", "split", "group_id", "sha256"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain: {sorted(required)}")
        rows = list(reader)
    image_ids = set()
    for row in rows:
        label = int(row["class_id"])
        if not 0 <= label < len(CLASS_NAMES) or CLASS_NAMES[label] != row["class_name"]:
            raise ValueError(f"Invalid class mapping for image {row['image_id']}")
        if row["split"] not in SPLITS:
            raise ValueError(f"Unsupported split: {row['split']}")
        if row["image_id"] in image_ids:
            raise ValueError(f"Duplicate image_id: {row['image_id']}")
        image_ids.add(row["image_id"])
        if not (path.parent / row["image_path"]).is_file():
            raise FileNotFoundError(path.parent / row["image_path"])
    splits = {s: [r for r in rows if r["split"] == s] for s in SPLITS}
    for s in ("train", "val"):
        if {int(r["class_id"]) for r in splits[s]} != set(range(len(CLASS_NAMES))):
            raise ValueError(f"All six classes must be represented in the {s} split.")
    # The same photo or near-duplicate group must never sit in two splits.
    for field in ("image_path", "group_id", "sha256"):
        for a in SPLITS:
            for b in SPLITS:
                if a < b:
                    overlap = {r[field] for r in splits[a]} & {r[field] for r in splits[b]}
                    if overlap:
                        raise ValueError(f"{a}/{b} leakage in {field}: {len(overlap)} values")
    return splits


class CachedImages(Dataset):
    """Decodes every photo once; training images are kept small and augmented on the fly."""

    def __init__(self, rows: list[dict], root: Path, train: bool, size: int = 224, crop_scale_min: float = 0.6):
        self.labels = [int(r["class_id"]) for r in rows]
        self.train = train
        if train:
            self.transform = make_train_transform(size, crop_scale_min)
            self.images = []
            keep = max(448, 2 * size)  # enough pixels for small random crops
            for r in rows:
                image = open_image(root / r["image_path"])
                image.thumbnail((keep, keep))
                self.images.append(image)
        else:
            transform = make_transform(size)
            self.images = [transform(open_image(root / r["image_path"])) for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index]
        return (self.transform(image) if self.train else image), self.labels[index]


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    logits, labels = [], []
    for images, targets in loader:
        logits.append(model(images.to(device)).cpu())
        labels.append(targets)
    return torch.cat(logits), torch.cat(labels)


def classification_metrics(labels, logits, split_name):
    probabilities = logits.softmax(1)
    predictions = probabilities.argmax(1)
    k = len(CLASS_NAMES)
    confusion = torch.bincount(labels * k + predictions, minlength=k * k).reshape(k, k)
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted = int(confusion[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    species_probs, condition_probs = marginal_probabilities(probabilities)
    condition_pred = condition_probs.argmax(1)
    rotten = labels % 2 == 1
    return {
        "split": split_name,
        "sample_count": len(labels),
        "loss": nn.functional.cross_entropy(logits, labels).item(),
        "accuracy": predictions.eq(labels).float().mean().item(),
        "macro_f1": sum(v["f1"] for v in per_class.values()) / k,
        "species_accuracy": species_probs.argmax(1).eq(labels // 2).float().mean().item(),
        "condition_accuracy": condition_pred.eq(labels % 2).float().mean().item(),
        "rotten_recall": condition_pred[rotten].eq(1).float().mean().item(),
        "fresh_recall": condition_pred[~rotten].eq(0).float().mean().item(),
        "species_condition_metric_method": "argmax of summed joint softmax probabilities",
        "class_names": CLASS_NAMES,
        "confusion_matrix_orientation": "rows=true, columns=predicted",
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
        "scores_are_calibrated": False,
    }


def write_predictions(path, rows, logits):
    probabilities = logits.softmax(1)
    species_probs, condition_probs = marginal_probabilities(probabilities)
    probability_columns = [f"prob_{name}" for name in CLASS_NAMES]
    fields = [
        "image_id", "image_path", "true_class", "predicted_class", "correct", "joint_score_uncalibrated",
        "predicted_species", "predicted_condition",
        "species_marginal_prediction", "condition_marginal_prediction",
    ] + probability_columns
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            predicted = int(probabilities[i].argmax())
            output = {
                "image_id": row["image_id"], "image_path": row["image_path"],
                "true_class": row["class_name"], "predicted_class": CLASS_NAMES[predicted],
                "correct": CLASS_NAMES[predicted] == row["class_name"],
                "joint_score_uncalibrated": round(float(probabilities[i, predicted]), 6),
                "predicted_species": SPECIES_NAMES[predicted // 2],
                "predicted_condition": CONDITION_NAMES[predicted % 2],
                "species_marginal_prediction": SPECIES_NAMES[int(species_probs[i].argmax())],
                "condition_marginal_prediction": CONDITION_NAMES[int(condition_probs[i].argmax())],
            }
            output.update({c: round(float(probabilities[i, j]), 6) for j, c in enumerate(probability_columns)})
            writer.writerow(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--arch", choices=["resnet18", "efficientnet_b0"], default="efficientnet_b0")
    parser.add_argument("--mode", choices=["linear", "finetune"], default="finetune")
    parser.add_argument("--pretrained", type=Path, help="Local official ImageNet state_dict; avoids downloads.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Classifier learning rate.")
    parser.add_argument("--backbone-lr", type=float, default=2e-4, help="Backbone learning rate (finetune).")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--no-augment", action="store_true", help="Train on plain resized photos.")
    parser.add_argument("--image-size", type=int, default=224, help="Square input size in pixels.")
    parser.add_argument("--pooling", choices=["avg", "avgmax"], default="avg",
                        help="Global pooling before the classifier; avgmax keeps the strongest local evidence.")
    parser.add_argument("--crop-scale-min", type=float, default=0.6,
                        help="Smallest random crop, as a fraction of the photo area (augmentation).")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1:
        raise ValueError("epochs, patience, batch-size and threads must be positive.")
    set_seed(args.seed, args.threads)
    device = resolve_device(args.device)
    manifest = args.manifest.expanduser().resolve()
    splits = read_manifest(manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    model, pretrained_source = create_model(args.arch, args.pretrained, args.pooling)
    head = classifier_layer(model, args.arch)
    head_ids = {id(p) for p in head.parameters()}
    head_prefix = next(n for n, m in model.named_modules() if m is head) + "."
    backbone_checksum_initial = state_checksum(model.state_dict(), exclude_prefixes=(head_prefix,))
    head_initial = {k: v.detach().clone() for k, v in head.state_dict().items()}
    model.to(device)

    augment = not args.no_augment
    linear = args.mode == "linear"
    for p in model.parameters():
        p.requires_grad_(not linear or id(p) in head_ids)

    eval_loaders = {
        s: DataLoader(CachedImages(splits[s], manifest.parent, train=False, size=args.image_size), batch_size=args.batch_size)
        for s in ("val", "test") if splits[s]
    }
    if linear and not augment:
        # Frozen backbone without augmentation: extract the features once (the 1st-round method).
        set_classifier_layer(model, args.arch, nn.Identity())
        feature_loader = DataLoader(CachedImages(splits["train"], manifest.parent, train=False, size=args.image_size),
                                    batch_size=args.batch_size)
        features, labels = predict_logits(model, feature_loader, device)
        set_classifier_layer(model, args.arch, head)
        train_loader = DataLoader(TensorDataset(features, labels),
                                  batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed))
        train_on_features = True
    else:
        train_loader = DataLoader(CachedImages(splits["train"], manifest.parent, train=augment,
                                                size=args.image_size, crop_scale_min=args.crop_scale_min),
                                  batch_size=args.batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(args.seed))
        train_on_features = False

    if linear:
        optimizer = torch.optim.Adam(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
    else:
        backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": args.backbone_lr},
            {"params": list(head.parameters()), "lr": args.lr},
        ], weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    setup_seconds = time.perf_counter() - started

    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    with (args.out / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_accuracy",
                                                    "val_loss", "val_accuracy", "val_macro_f1", "seconds"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            if linear:
                model.eval()  # BatchNorm statistics stay frozen
                head.train()
            else:
                model.train()
            running_loss, running_correct, seen = 0.0, 0, 0
            for batch, targets in train_loader:
                batch, targets = batch.to(device), targets.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = head(batch) if train_on_features else model(batch)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(targets)
                running_correct += int(logits.argmax(1).eq(targets).sum())
                seen += len(targets)
            if scheduler:
                scheduler.step()
            val_logits, val_labels = predict_logits(model, eval_loaders["val"], device)
            val = classification_metrics(val_labels, val_logits, "val")
            entry = {
                "epoch": epoch, "train_loss": running_loss / seen, "train_accuracy": running_correct / seen,
                "val_loss": val["loss"], "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                "seconds": round(time.perf_counter() - epoch_start, 1),
            }
            history.append(entry)
            writer.writerow(entry)
            handle.flush()
            print(json.dumps(entry), flush=True)
            if val["loss"] < best_loss:
                best_loss, best_epoch, stale = val["loss"], epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
                    break

    model.load_state_dict(best_state)
    results, logits_by_split = {}, {}
    for split_name, loader in eval_loaders.items():
        logits, labels = predict_logits(model, loader, device)
        logits_by_split[split_name] = logits
        results[split_name] = classification_metrics(labels, logits, split_name)
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    head_final = classifier_layer(model, args.arch).state_dict()
    verification = {
        "classifier_parameter_delta_l2": float(torch.cat(
            [(head_final[k].cpu() - v.cpu()).flatten() for k, v in head_initial.items()]).norm()),
        "backbone_sha256_initial": backbone_checksum_initial,
        "backbone_sha256_final": state_checksum(final_state, exclude_prefixes=(head_prefix,)),
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_parameter_count": sum(p.numel() for p in model.parameters()),
    }
    verification["backbone_changed"] = verification["backbone_sha256_initial"] != verification["backbone_sha256_final"]
    if linear and verification["backbone_changed"]:
        raise RuntimeError("Linear mode must not change the frozen backbone.")
    counts = {s: {n: Counter(r["class_name"] for r in rows)[n] for n in CLASS_NAMES} for s, rows in splits.items()}
    metadata = {
        "format_version": 2, "architecture": args.arch, "training_mode": args.mode, "pooling": args.pooling,
        "class_names": CLASS_NAMES, "species_names": SPECIES_NAMES, "condition_names": CONDITION_NAMES,
        "preprocess": preprocess_for(args.image_size), "seed": args.seed, "best_epoch": best_epoch,
        "epochs_completed": len(history), "counts": counts,
        "selection": "lowest validation cross-entropy; the test split is scored once after selection",
        "metrics": results, "training_verification": verification,
        "hyperparameters": {
            "optimizer": "Adam" if linear else "AdamW", "lr": args.lr,
            "backbone_lr": None if linear else args.backbone_lr, "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing, "batch_size": args.batch_size,
            "max_epochs": args.epochs, "early_stopping_patience": args.patience,
            "scheduler": None if linear else "cosine", "augmentation": augment,
            "image_size": args.image_size, "crop_scale_min": args.crop_scale_min if augment else None,
            "device": args.device, "threads": args.threads,
        },
        "pretrained_source": pretrained_source,
        "runtime": {"torch_version": str(torch.__version__)},
        "timing_seconds": {"setup": round(setup_seconds, 1), "total": round(time.perf_counter() - started, 1)},
    }
    torch.save({**metadata, "state_dict": final_state}, args.out / "model.pth")
    with (args.out / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
    for split_name, logits in logits_by_split.items():
        write_predictions(args.out / f"{split_name}_predictions.csv", splits[split_name], logits)
    summary = {s: {k: round(v[k], 4) for k in ("accuracy", "macro_f1", "species_accuracy", "condition_accuracy")}
               for s, v in results.items()}
    print(json.dumps({"saved_to": str(args.out.resolve()), "best_epoch": best_epoch, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
