"""Classify a photo or a directory using a trained checkpoint (vegetable species x fresh/rotten)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

from model_utils import (
    CONDITION_NAMES, IMAGE_EXTENSIONS, SPECIES_NAMES, UNKNOWN_MESSAGE,
    calibrated_probabilities, input_size, is_unknown, load_model, make_transform, marginal_probabilities,
    read_image, resolve_device, unknown_score,
)
KOREAN = {"cucumber": "오이", "potato": "감자", "tomato": "토마토", "bellpepper": "피망", "carrot": "당근",
          "fresh": "정상", "rotten": "비정상"}


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
                        help="Mark a photo as uncertain when the top class probability is below this value.")
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
    class_names = checkpoint["class_names"]
    probability_columns = [f"prob_{name}" for name in class_names]
    calibrated = "calibration" in checkpoint
    fields = [
        "image", "known", "message", "predicted_class", "species", "condition", "probability", "uncertain",
        "species_marginal_prediction", "species_probability",
        "condition_marginal_prediction", "condition_probability", "unknown_score", "error",
    ] + probability_columns
    if not calibrated:
        print("Note: this checkpoint has no calibration; run src/calibrate.py to get calibrated probabilities.",
              file=sys.stderr)
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
                    logits = model(image).cpu()
                probabilities = calibrated_probabilities(logits, checkpoint)
                index = int(probabilities[0].argmax())
                species_probs, condition_probs = marginal_probabilities(probabilities)
                species_index = int(species_probs[0].argmax())
                condition_index = int(condition_probs[0].argmax())
                unknown = bool(is_unknown(logits, checkpoint)[0])
                row.update({
                    "known": "no" if unknown else "yes",
                    "message": UNKNOWN_MESSAGE if unknown else "",
                    "predicted_class": "unknown" if unknown else class_names[index],
                    "species": "" if unknown else SPECIES_NAMES[index // 2],
                    "condition": "" if unknown else CONDITION_NAMES[index % 2],
                    "probability": round(float(probabilities[0, index]), 4),
                    "uncertain": "yes" if float(probabilities[0, index]) < args.uncertain_below else "no",
                    "species_marginal_prediction": SPECIES_NAMES[species_index],
                    "species_probability": round(float(species_probs[0, species_index]), 4),
                    "condition_marginal_prediction": CONDITION_NAMES[condition_index],
                    "condition_probability": round(float(condition_probs[0, condition_index]), 4),
                    "unknown_score": round(float(unknown_score(logits)[0]), 4),
                    "error": "",
                })
                row.update({column: round(float(probabilities[0, i]), 4) for i, column in enumerate(probability_columns)})
                success += 1
                if unknown:
                    print(f"{path.name}: {UNKNOWN_MESSAGE}")
                else:
                    flag = " [uncertain]" if row["uncertain"] == "yes" else ""
                    word = "probability" if calibrated else "uncalibrated score"
                    print(f"{path.name}: {class_names[index]} ({KOREAN[SPECIES_NAMES[index // 2]]} "
                          f"{KOREAN[CONDITION_NAMES[index % 2]]}, {word} {100 * row['probability']:.1f}%){flag}")
            except (OSError, ValueError, RuntimeError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"Could not classify {path}: {error}", file=sys.stderr)
            writer.writerow(row)
    print(f"Saved {success}/{len(photos)} successful predictions to {args.output.resolve()}")
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
