"""Draw learning curves, test confusion matrices and the model comparison from results/*/ (needs matplotlib)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, INK_2, GRID, ACCENT, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#9a9994"
BLUES = ["#f7f9fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
LABELS = {"resnet18_linear": "ResNet18, last layer only (1st-round method)",
          "resnet18_finetune": "ResNet18, fine-tuned + augmentation",
          "efficientnet_b0_finetune": "EfficientNet-B0, fine-tuned + augmentation"}


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def learning_curves(run: Path, meta: dict):
    rows = list(csv.DictReader((run / "history.csv").open(encoding="utf-8")))
    epochs = [int(r["epoch"]) for r in rows]
    best = meta["best_epoch"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax in (left, right):
        style(ax)
        ax.axvline(best, color=MUTED, linewidth=1, linestyle="--")
        ax.set_xlabel("epoch", color=INK_2)
    left.plot(epochs, [float(r["train_loss"]) for r in rows], color=MUTED, linewidth=2, label="train")
    left.plot(epochs, [float(r["val_loss"]) for r in rows], color=ACCENT, linewidth=2, label="validation")
    left.set_title("Loss (cross-entropy)", color=INK, fontsize=11, loc="left")
    left.legend(frameon=False, fontsize=9, labelcolor=INK_2)
    right.plot(epochs, [100 * float(r["val_accuracy"]) for r in rows], color=ACCENT, linewidth=2)
    right.set_title("Validation accuracy (%)", color=INK, fontsize=11, loc="left")
    right.annotate(f"selected epoch {best}", (best, right.get_ylim()[0]), xytext=(4, 4),
                   textcoords="offset points", color=INK_2, fontsize=8)
    fig.suptitle(LABELS.get(run.name, run.name), color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(run / "learning_curves.png", dpi=150, facecolor="white")
    plt.close(fig)


def confusion(run: Path, meta: dict, split: str = "test"):
    m = meta["metrics"][split]
    matrix, names = m["confusion_matrix"], [n.replace("_", "\n") for n in m["class_names"]]
    peak = max(max(r) for r in matrix)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", BLUES)
    ax.imshow(matrix, cmap=cmap, vmin=0, vmax=peak)
    for i, row in enumerate(matrix):
        for j, v in enumerate(row):
            if v:
                ax.text(j, i, v, ha="center", va="center", fontsize=10,
                        color="white" if v > 0.55 * peak else INK)
    ax.set_xticks(range(len(names)), names, fontsize=8, color=INK_2)
    ax.set_yticks(range(len(names)), names, fontsize=8, color=INK_2)
    ax.set_xlabel("predicted", color=INK_2)
    ax.set_ylabel("true", color=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"{split.capitalize()} set: {round(100 * m['accuracy'], 1)}% correct "
                 f"({sum(matrix[i][i] for i in range(len(matrix)))}/{m['sample_count']})",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(run / f"confusion_matrix_{split}.png", dpi=150, facecolor="white")
    plt.close(fig)


def comparison(runs: list[Path], metas: list[dict], out: Path, highlight: str):
    metrics = [("accuracy", "6 classes"), ("species_accuracy", "vegetable"), ("condition_accuracy", "fresh / rotten")]
    order = ["resnet18_linear", "resnet18_finetune", "efficientnet_b0_finetune"]
    idx = sorted(range(len(runs)), key=lambda k: order.index(runs[k].name) if runs[k].name in order else 99)
    shades = {"resnet18_linear": "#d6d5d0", "resnet18_finetune": MUTED}
    fig, ax = plt.subplots(figsize=(9, 4.4))
    style(ax)
    width = 0.8 / len(runs)
    for slot, k in enumerate(idx):
        color = ACCENT if runs[k].name == highlight else shades.get(runs[k].name, MUTED)
        values = [100 * metas[k]["metrics"]["test"][key] for key, _ in metrics]
        xs = [g + (slot - (len(runs) - 1) / 2) * width for g in range(len(metrics))]
        bars = ax.bar(xs, values, width * 0.92, color=color, label=LABELS.get(runs[k].name, runs[k].name))
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=8, color=INK_2)
    ax.set_xticks(range(len(metrics)), [label for _, label in metrics], color=INK_2)
    ax.set_ylim(0, 105)
    ax.set_ylabel("test accuracy (%)", color=INK_2)
    acc = {r.name: 100 * m["metrics"]["test"]["accuracy"] for r, m in zip(runs, metas)}
    tuned = [v for n, v in acc.items() if n.endswith("finetune")]
    title = (f"Fine-tuning lifts 6-class test accuracy from {acc.get('resnet18_linear', 0):.1f}% "
             f"to {min(tuned):.1f}-{max(tuned):.1f}%" if "resnet18_linear" in acc and tuned else "Test accuracy")
    ax.set_title(title + "  (same 180 test photos)", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("results"))
    parser.add_argument("--highlight", default="efficientnet_b0_finetune")
    parser.add_argument("--out", type=Path, default=Path("results/model_comparison.png"))
    args = parser.parse_args()
    runs, metas = [], []
    for p in sorted(args.runs.iterdir()):
        meta = json.loads((p / "metrics.json").read_text(encoding="utf-8")) if (p / "metrics.json").is_file() else {}
        if "test" in meta.get("metrics", {}):  # skip the 1st-round record (no test split)
            runs.append(p)
            metas.append(meta)
    for run, meta in zip(runs, metas):
        learning_curves(run, meta)
        confusion(run, meta, "test")
    comparison(runs, metas, args.out, args.highlight)
    print("plots written for", [r.name for r in runs])


if __name__ == "__main__":
    main()
