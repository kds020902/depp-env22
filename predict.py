"""Classify a photo or a directory using the trained six-class checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

from model_utils import (
    CLASS_NAMES, CONDITION_NAMES, IMAGE_EXTENSIONS, SPECIES_NAMES,
    input_size, load_model, make_transform, marginal_probabilities, read_image,
    resolve_device,
)


def find_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Photo path or recursively searched directory.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--uncertain-below", type=float, default=0.5,
                        help="Mark a photo as uncertain when the top class score is below this value.")
    args = parser.parse_args()
    if args.threads < 1:
        raise ValueError("threads must be positive.")
    torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.model.expanduser().resolve(), device)
    photos = find_images(args.input.expanduser().resolve())
    if not photos:
        raise ValueError("No supported image files were found.")
    transform = make_transform(input_size(checkpoint))
    probability_columns = [f"prob_{name}" for name in CLASS_NAMES]
    fields = [
        "image", "predicted_class", "species", "condition", "joint_score_uncalibrated", "uncertain",
        "species_marginal_prediction", "species_score_uncalibrated",
        "condition_marginal_prediction", "condition_score_uncalibrated", "error",
    ] + probability_columns
    args.output.parent.mkdir(parents=True, exist_ok=True)
    success = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in photos:
            row = {"image": str(path)}
            try:
                image = read_image(path, transform).unsqueeze(0).to(device)
                with torch.inference_mode():
                    probabilities = model(image).softmax(1).cpu()
                index = int(probabilities[0].argmax())
                species_probs, condition_probs = marginal_probabilities(probabilities)
                species_index = int(species_probs[0].argmax())
                condition_index = int(condition_probs[0].argmax())
                row.update({
                    "predicted_class": CLASS_NAMES[index],
                    "species": SPECIES_NAMES[index // 2],
                    "condition": CONDITION_NAMES[index % 2],
                    "joint_score_uncalibrated": float(probabilities[0, index]),
                    "uncertain": "yes" if float(probabilities[0, index]) < args.uncertain_below else "no",
                    "species_marginal_prediction": SPECIES_NAMES[species_index],
                    "species_score_uncalibrated": float(species_probs[0, species_index]),
                    "condition_marginal_prediction": CONDITION_NAMES[condition_index],
                    "condition_score_uncalibrated": float(condition_probs[0, condition_index]),
                    "error": "",
                })
                row.update({column: float(probabilities[0, i]) for i, column in enumerate(probability_columns)})
                success += 1
                flag = " [uncertain]" if row["uncertain"] == "yes" else ""
                print(f"{path.name}: {row['predicted_class']} (uncalibrated score {row['joint_score_uncalibrated']:.4f}){flag}")
            except (OSError, ValueError, RuntimeError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"Could not classify {path}: {error}", file=sys.stderr)
            writer.writerow(row)
    print(f"Saved {success}/{len(photos)} successful predictions to {args.output.resolve()}")
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
