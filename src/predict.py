"""Classify a photo or a directory using a trained checkpoint (vegetable species x fresh/rotten)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

from model_utils import (
    CONDITION_NAMES, IMAGE_EXTENSIONS, SPECIES_NAMES, UNKNOWN_MESSAGE,
    calibrated_probabilities, forward_with_rot_map, input_size, is_unknown, load_model, make_transform,
    marginal_probabilities, open_image, resolve_device, rot_map_overlay, unknown_score,
)
RETAKE_MESSAGE = "확신이 낮습니다. 채소 하나만, 밝은 곳에서, 상한 부분이 보이게 가까이 다시 찍어 보세요."
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
    parser.add_argument("--retake-below", type=float, default=0.9,
                        help="Suggest taking the photo again when the top class probability is below this value "
                             "(on Valid, 97.6%% of photos at 90%% or more were right, 73%% of those below).")
    parser.add_argument("--clip-hint", action="store_true",
                        help="For photos turned away, also name the closest thing with CLIP (needs open_clip_torch; "
                             "downloads about 600 MB on first use). It does not change which photos are turned away.")
    parser.add_argument("--heatmap", type=Path,
                        help="Folder for heatmaps: each photo next to a copy where the parts the model saw as rotten "
                             "are coloured red (stronger red = stronger evidence).")
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
        "image", "known", "message", "closest_guess", "retake", "predicted_class", "species", "condition",
        "probability", "uncertain",
        "species_marginal_prediction", "species_probability",
        "condition_marginal_prediction", "condition_probability", "unknown_score", "rot_evidence", "heatmap", "error",
    ] + probability_columns
    if not calibrated:
        print("Note: this checkpoint has no calibration; run src/calibrate.py to get calibrated probabilities.",
              file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rot_scale = checkpoint.get("rot_map", {}).get("scale")
    if args.heatmap:
        args.heatmap.mkdir(parents=True, exist_ok=True)
    hint = None
    if args.clip_hint:
        from clip_hint import ClipHint
        try:
            hint = ClipHint(args.device)
        except ImportError:
            raise SystemExit("--clip-hint needs open_clip: python -m pip install open_clip_torch") from None
    success = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in photos:
            row = {"image": str(path)}
            try:
                photo = open_image(path)
                with torch.inference_mode():
                    logits, rot_map = forward_with_rot_map(model, checkpoint, transform(photo).unsqueeze(0).to(device))
                logits, rot_map = logits.cpu(), rot_map[0].cpu()
                scale = rot_scale or float(rot_map.max()) or 1.0
                probabilities = calibrated_probabilities(logits, checkpoint)
                index = int(probabilities[0].argmax())
                species_probs, condition_probs = marginal_probabilities(probabilities)
                species_index = int(species_probs[0].argmax())
                condition_index = int(condition_probs[0].argmax())
                unknown = bool(is_unknown(logits, checkpoint)[0])
                retake = unknown or float(probabilities[0, index]) < args.retake_below
                message = UNKNOWN_MESSAGE if unknown else ""
                closest = ""
                if unknown and hint is not None:
                    closest, sentence = hint.message(photo)
                    message = f"{UNKNOWN_MESSAGE} ({sentence})"
                elif retake:
                    message = RETAKE_MESSAGE
                row.update({
                    "known": "no" if unknown else "yes",
                    "message": message,
                    "closest_guess": closest,
                    "retake": "yes" if retake else "no",
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
                    "rot_evidence": "" if unknown else round(float(rot_map.max()) / scale, 3),
                    "heatmap": "",
                    "error": "",
                })
                if args.heatmap and not unknown:
                    heatmap_path = args.heatmap / f"{path.stem}_heatmap.jpg"
                    rot_map_overlay(photo, rot_map, scale).save(heatmap_path, quality=90)
                    row["heatmap"] = str(heatmap_path)
                row.update({column: round(float(probabilities[0, i]), 4) for i, column in enumerate(probability_columns)})
                success += 1
                if unknown:
                    print(f"{path.name}: {message}")
                else:
                    flag = " [uncertain]" if row["uncertain"] == "yes" else ""
                    word = "probability" if calibrated else "uncalibrated score"
                    advice = f" - {RETAKE_MESSAGE}" if retake else ""
                    print(f"{path.name}: {class_names[index]} ({KOREAN[SPECIES_NAMES[index // 2]]} "
                          f"{KOREAN[CONDITION_NAMES[index % 2]]}, {word} {100 * row['probability']:.1f}%){flag}{advice}")
            except (OSError, ValueError, RuntimeError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
                print(f"Could not classify {path}: {error}", file=sys.stderr)
            writer.writerow(row)
    print(f"Saved {success}/{len(photos)} successful predictions to {args.output.resolve()}")
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
