"""Train a six-class linear ResNet18 head on fixed ImageNet features."""

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
    CLASS_NAMES, CONDITION_NAMES, PREPROCESS, SPECIES_NAMES,
    create_model, make_transform, marginal_probabilities, read_image,
    resolve_device, set_seed, state_checksum,
)


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict], root: Path):
        self.rows = rows
        self.root = root
        self.transform = make_transform()

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image = read_image(self.root / row["image_path"], self.transform)
        return image, int(row["class_id"])


def read_manifest(path: Path) -> tuple[list[dict], list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "image_path", "class_name", "class_id", "split"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest must contain: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError("The manifest is empty.")
    image_ids = set()
    for row in rows:
        label = int(row["class_id"])
        if not 0 <= label < len(CLASS_NAMES) or CLASS_NAMES[label] != row["class_name"]:
            raise ValueError(f"Invalid class mapping for image {row['image_id']}")
        if row["split"] not in {"train", "val"}:
            raise ValueError(f"Unsupported split: {row['split']}")
        if row["image_id"] in image_ids:
            raise ValueError(f"Duplicate image_id: {row['image_id']}")
        image_ids.add(row["image_id"])
        if not (path.parent / row["image_path"]).is_file():
            raise FileNotFoundError(path.parent / row["image_path"])
    train = [row for row in rows if row["split"] == "train"]
    val = [row for row in rows if row["split"] == "val"]
    if not train or not val:
        raise ValueError("Both train and val must contain images.")
    if {int(row["class_id"]) for row in train} != set(range(len(CLASS_NAMES))):
        raise ValueError("All six classes must be represented in the training split.")
    for field in ("image_path", "group_id", "sha256"):
        train_values = {row[field] for row in train if row.get(field)}
        val_values = {row[field] for row in val if row.get(field)}
        overlap = train_values & val_values
        if overlap:
            raise ValueError(f"Train/val leakage in {field}: {len(overlap)} overlapping values")
    return train, val


@torch.no_grad()
def extract_features(backbone, loader, device, split):
    backbone.eval()
    all_features, all_labels = [], []
    total = len(loader.dataset)
    processed = 0
    for images, labels in loader:
        features = backbone(images.to(device)).flatten(1)
        all_features.append(features.cpu())
        all_labels.append(labels)
        processed += len(labels)
        if processed % (loader.batch_size * 10) == 0 or processed == total:
            print(f"features {split}: {processed}/{total}", flush=True)
    return torch.cat(all_features), torch.cat(all_labels)


@torch.no_grad()
def evaluate_head(head, features, labels, device, batch_size):
    head.eval()
    outputs = []
    for batch in features.split(batch_size):
        outputs.append(head(batch.to(device)).cpu())
    logits = torch.cat(outputs)
    loss = nn.functional.cross_entropy(logits, labels).item()
    accuracy = logits.argmax(1).eq(labels).float().mean().item()
    return loss, accuracy, logits


def classification_metrics(labels, logits):
    probabilities = logits.softmax(1)
    predictions = probabilities.argmax(1)
    confusion = torch.bincount(
        labels * len(CLASS_NAMES) + predictions,
        minlength=len(CLASS_NAMES) ** 2,
    ).reshape(len(CLASS_NAMES), len(CLASS_NAMES))
    per_class = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support,
        }
    species_probs, condition_probs = marginal_probabilities(probabilities)
    return {
        "evaluation_split": "validation (also used for early stopping; not an independent test set)",
        "sample_count": len(labels),
        "loss": nn.functional.cross_entropy(logits, labels).item(),
        "accuracy": predictions.eq(labels).float().mean().item(),
        "macro_f1": sum(item["f1"] for item in per_class.values()) / len(CLASS_NAMES),
        "balanced_accuracy": sum(item["recall"] for item in per_class.values()) / len(CLASS_NAMES),
        "species_accuracy": species_probs.argmax(1).eq(labels // 2).float().mean().item(),
        "condition_accuracy": condition_probs.argmax(1).eq(labels % 2).float().mean().item(),
        "species_condition_metric_method": "argmax of summed joint softmax probabilities",
        "joint_prediction_species_accuracy": (predictions // 2).eq(labels // 2).float().mean().item(),
        "joint_prediction_condition_accuracy": (predictions % 2).eq(labels % 2).float().mean().item(),
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
        "image_id", "image_path", "true_class", "predicted_class", "joint_score_uncalibrated",
        "true_species", "true_condition", "predicted_species", "predicted_condition",
        "species_marginal_prediction", "species_score_uncalibrated",
        "condition_marginal_prediction", "condition_score_uncalibrated",
    ] + probability_columns
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            predicted = int(probabilities[i].argmax())
            truth = int(row["class_id"])
            species = int(species_probs[i].argmax())
            condition = int(condition_probs[i].argmax())
            output = {
                "image_id": row["image_id"], "image_path": row["image_path"],
                "true_class": CLASS_NAMES[truth], "predicted_class": CLASS_NAMES[predicted],
                "joint_score_uncalibrated": float(probabilities[i, predicted]),
                "true_species": SPECIES_NAMES[truth // 2], "true_condition": CONDITION_NAMES[truth % 2],
                "predicted_species": SPECIES_NAMES[predicted // 2],
                "predicted_condition": CONDITION_NAMES[predicted % 2],
                "species_marginal_prediction": SPECIES_NAMES[species],
                "species_score_uncalibrated": float(species_probs[i, species]),
                "condition_marginal_prediction": CONDITION_NAMES[condition],
                "condition_score_uncalibrated": float(condition_probs[i, condition]),
            }
            output.update({column: float(probabilities[i, j]) for j, column in enumerate(probability_columns)})
            writer.writerow(output)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, help="Local official ResNet18 ImageNet state_dict; avoids downloads.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1:
        raise ValueError("epochs, patience, batch-size and threads must be positive.")
    set_seed(args.seed, args.threads)
    device = resolve_device(args.device)
    manifest = args.manifest.expanduser().resolve()
    train_rows, val_rows = read_manifest(manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, pretrained_source = create_model(args.pretrained)
    backbone_checksum_initial = state_checksum(model.state_dict(), exclude_fc=True)
    classifier_initial = {name: value.detach().cpu().clone() for name, value in model.fc.state_dict().items()}
    model.to(device).eval()
    backbone = nn.Sequential(*list(model.children())[:-1]).to(device).eval()
    features = {}
    for split, rows in (("train", train_rows), ("val", val_rows)):
        loader = DataLoader(
            ManifestDataset(rows, manifest.parent), batch_size=args.batch_size,
            shuffle=False, num_workers=0,
        )
        features[split] = extract_features(backbone, loader, device, split)
    feature_seconds = time.perf_counter() - started
    train_features, train_labels = features["train"]
    val_features, val_labels = features["val"]
    train_loader = DataLoader(
        TensorDataset(train_features, train_labels), batch_size=args.batch_size,
        shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    initial_val_loss, initial_val_accuracy, _ = evaluate_head(
        model.fc, val_features, val_labels, device, args.batch_size,
    )
    best_loss, best_epoch, best_state, stale = float("inf"), 0, None, 0
    history = []
    history_path = args.out / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.fc.train()  # Never call model.train(): the backbone/BN stay frozen.
            running_loss, running_correct, seen = 0.0, 0, 0
            for batch, labels in train_loader:
                batch, labels = batch.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model.fc(batch)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * len(labels)
                running_correct += int(logits.argmax(1).eq(labels).sum())
                seen += len(labels)
            val_loss, val_accuracy, _ = evaluate_head(model.fc, val_features, val_labels, device, args.batch_size)
            entry = {
                "epoch": epoch, "train_loss": running_loss / seen,
                "train_accuracy": running_correct / seen,
                "val_loss": val_loss, "val_accuracy": val_accuracy,
            }
            history.append(entry)
            writer.writerow(entry)
            handle.flush()
            print(json.dumps(entry), flush=True)
            if val_loss < best_loss:
                best_loss, best_epoch, stale = val_loss, epoch, 0
                best_state = {name: value.detach().cpu().clone() for name, value in model.fc.state_dict().items()}
            else:
                stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}", flush=True)
                break
    model.fc.load_state_dict(best_state)
    model.eval()
    _, _, val_logits = evaluate_head(model.fc, val_features, val_labels, device, args.batch_size)
    metrics = classification_metrics(val_labels, val_logits)
    final_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    backbone_checksum_final = state_checksum(final_state, exclude_fc=True)
    deltas = torch.cat([
        (model.fc.state_dict()[name].detach().cpu() - before).flatten()
        for name, before in classifier_initial.items()
    ])
    verification = {
        "classifier_parameter_delta_l2": float(deltas.norm()),
        "classifier_parameter_delta_max_abs": float(deltas.abs().max()),
        "classifier_changed": bool(torch.count_nonzero(deltas)),
        "classifier_sha256_initial": state_checksum(classifier_initial),
        "classifier_sha256_final": state_checksum(model.fc.state_dict()),
        "frozen_backbone_sha256_initial": backbone_checksum_initial,
        "frozen_backbone_sha256_final": backbone_checksum_final,
        "frozen_backbone_unchanged": backbone_checksum_initial == backbone_checksum_final,
        "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    if not verification["classifier_changed"] or not verification["frozen_backbone_unchanged"]:
        raise RuntimeError(f"Training invariant failed: {verification}")
    counts = {
        split: {name: Counter(row["class_name"] for row in rows)[name] for name in CLASS_NAMES}
        for split, rows in (("train", train_rows), ("val", val_rows))
    }
    metadata = {
        "format_version": 1, "architecture": "resnet18",
        "training_method": "frozen ImageNet backbone with trained six-class linear head",
        "class_names": CLASS_NAMES, "species_names": SPECIES_NAMES,
        "condition_names": CONDITION_NAMES, "preprocess": PREPROCESS,
        "seed": args.seed, "best_epoch": best_epoch, "epochs_completed": len(history),
        "counts": counts, "metrics": metrics, "training_verification": verification,
        "hyperparameters": {
            "optimizer": "Adam", "lr": args.lr, "weight_decay": args.weight_decay,
            "batch_size": args.batch_size, "max_epochs": args.epochs,
            "early_stopping_patience": args.patience, "selection_metric": "validation_cross_entropy",
            "threads": args.threads, "device": args.device,
            "num_workers": 0, "augmentation": "none; deterministic features extracted once",
        },
        "initial_validation": {"loss": initial_val_loss, "accuracy": initial_val_accuracy},
        "pretrained_source": pretrained_source,
        "runtime": {"torch_version": str(torch.__version__)},
        "timing_seconds": {
            "feature_extraction": feature_seconds,
            "total": time.perf_counter() - started,
        },
    }
    checkpoint = {**metadata, "state_dict": final_state}
    torch.save(checkpoint, args.out / "model.pth")
    with (args.out / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False, allow_nan=False)
    write_predictions(args.out / "val_predictions.csv", val_rows, val_logits)
    print(json.dumps({"saved_to": str(args.out.resolve()), "best_epoch": best_epoch, "metrics": metrics}, indent=2), flush=True)


if __name__ == "__main__":
    main()
