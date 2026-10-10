"""Fruit gate: a small MobileNetV3-Small that only answers "fruit or one of the five vegetables?".

    python experiments/fruit_gate/fruit_gate.py train --train gate_train.csv --val gate_val.csv --out runs/G42 --seed 42
    python experiments/fruit_gate/fruit_gate.py score --model runs/G42/gate.pth photo.jpg ...

The vegetable model (model.pth) is not touched. In use, a photo whose fruit probability is above the gate's threshold is
answered "학습된 데이터가 아닙니다"; every other photo goes to the vegetable model as before.
The threshold is fitted on the Valid vegetable photos only: the 99th percentile of their fruit probability (at least 0.5).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
from model_utils import make_train_transform, make_transform, open_image, set_seed  # noqa: E402

GATE_CLASSES = ["vegetable", "fruit"]


def build_gate(pretrained: bool = True) -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(GATE_CLASSES))
    return model


def load_gate(path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_gate(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint


def fruit_probability(model, images: torch.Tensor) -> torch.Tensor:
    return model(images).softmax(1)[:, 1]


class Photos(Dataset):
    def __init__(self, rows, train: bool):
        self.labels = [int(r["label"]) for r in rows]
        self.train = train
        if train:
            self.transform = make_train_transform(224, 0.6, 0.0)
            self.images = []
            for r in rows:
                image = open_image(r["path"])
                image.thumbnail((448, 448))
                self.images.append(image)
        else:
            transform = make_transform(224)
            self.images = [transform(open_image(r["path"])) for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (self.transform(self.images[i]) if self.train else self.images[i]), self.labels[i]


def read(path):
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, labels, loss = [], [], 0.0
    for x, y in loader:
        logits = model(x)
        loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        probs.append(logits.softmax(1)[:, 1]); labels.append(y)
    probs, labels = torch.cat(probs), torch.cat(labels)
    return probs, labels, loss / len(labels)


def train(args):
    set_seed(args.seed, args.threads)
    train_rows, val_rows = read(args.train), read(args.val)
    train_set, val_set = Photos(train_rows, True), Photos(val_rows, False)
    loader = DataLoader(train_set, batch_size=32, shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    val_loader = DataLoader(val_set, batch_size=64)
    model = build_gate()
    head = list(model.classifier.parameters())
    head_ids = {id(p) for p in head}
    optimizer = torch.optim.AdamW([{"params": [p for p in model.parameters() if id(p) not in head_ids], "lr": 1e-4},
                                   {"params": head, "lr": 1e-3}], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    counts = torch.bincount(torch.tensor(train_set.labels), minlength=2).float()
    criterion = nn.CrossEntropyLoss(weight=counts.sum() / (2 * counts), label_smoothing=0.1)
    args.out.mkdir(parents=True, exist_ok=True)
    best, best_state, stale, history = float("inf"), None, 0, []
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            criterion(model(x), y).backward()
            optimizer.step()
        scheduler.step()
        probs, labels, val_loss = predict(model, val_loader)
        acc = ((probs > 0.5).long() == labels).float().mean().item()
        entry = {"epoch": epoch, "val_loss": val_loss, "val_accuracy": acc, "seconds": round(time.perf_counter() - start, 1)}
        history.append(entry); print(json.dumps(entry), flush=True)
        if val_loss < best:
            best, stale, best_state = val_loss, 0, {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    probs, labels, val_loss = predict(model, val_loader)
    veg = probs[labels == 0]
    threshold = max(0.5, float(torch.quantile(veg, 0.99)))
    summary = {"val_loss": val_loss, "threshold": threshold,
               "val_vegetables_rejected": int((veg > threshold).sum()), "val_vegetables": int(len(veg)),
               "val_fruit_rejected": int((probs[labels == 1] > threshold).sum()), "val_fruit": int((labels == 1).sum()),
               "train_counts": {"vegetable": int(counts[0]), "fruit": int(counts[1])}, "epochs": len(history),
               "best_epoch": min(history, key=lambda h: h["val_loss"])["epoch"], "seed": args.seed}
    torch.save({"architecture": "mobilenet_v3_small", "classes": GATE_CLASSES, "threshold": threshold,
                "threshold_rule": "99th percentile of Valid vegetable fruit-probability, at least 0.5",
                "preprocess": "same as model.pth (224, ImageNet mean/std)", "summary": summary, "history": history,
                "state_dict": model.state_dict()}, args.out / "gate.pth")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def score(args):
    model, checkpoint = load_gate(args.model)
    transform = make_transform(224)
    with torch.inference_mode():
        for path in args.photos:
            p = float(fruit_probability(model, transform(open_image(path)).unsqueeze(0))[0])
            print(f"{path}\tfruit {p:.3f}\t{'fruit' if p > checkpoint['threshold'] else 'vegetable'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--train", type=Path, required=True)
    t.add_argument("--val", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--patience", type=int, default=5)
    t.add_argument("--threads", type=int, default=4)
    s = sub.add_parser("score")
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("photos", nargs="+")
    args = parser.parse_args()
    train(args) if args.command == "train" else score(args)


if __name__ == "__main__":
    main()
