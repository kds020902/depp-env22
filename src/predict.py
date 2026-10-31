"""Classify a photo or a directory using a trained checkpoint (vegetable species x fresh/rotten).

When the checkpoint has a CLIP head (model.pth does), the vegetable model and the CLIP classifier are combined and a photo
is turned away when either says it is not one of the trained vegetables (experiments/clip_classifier). This needs
`pip install open_clip_torch` and downloads CLIP (about 600 MB) on first use; --no-clip uses the vegetable model alone.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch

from model_utils import (
    CONDITION_NAMES, IMAGE_EXTENSIONS, NO_CONDITION_SPECIES, SPECIES_NAMES, UNKNOWN_MESSAGE,
    calibrated_probabilities, clip_head_logits, clip_unknown_score, combined_logits, combined_probabilities, forward_with_rot_maps,
    input_size, is_unknown, is_unknown_combined, load_model, make_transform, marginal_probabilities, open_image,
    resolve_device, rot_map_overlay, unknown_score,
)
from report import image_data_uri, write_report

RETAKE_MESSAGE = "확신이 낮습니다. 채소 하나만, 밝은 곳에서, 상한 부분이 보이게 가까이 다시 찍어 보세요."
KOREAN = {"cucumber": "오이", "potato": "감자", "tomato": "토마토", "bellpepper": "피망", "carrot": "당근",
          "onion": "양파", "sweetpotato": "고구마", "greenonion": "대파",
          "fresh": "정상", "rotten": "비정상", "not_judged": "상태 판정 안 함"}


def condition_of(index: int) -> str:
    """fresh / rotten for a class id; not_judged for a vegetable whose condition the model was not trained to judge."""
    return "not_judged" if SPECIES_NAMES[index // 2] in NO_CONDITION_SPECIES else CONDITION_NAMES[index % 2]


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
                             "(combined model on Valid: 98.9%% of photos at 90%% or more were right, 68%% of those below).")
    parser.add_argument("--no-clip", action="store_true",
                        help="Use the vegetable model alone, without CLIP (faster, no 600 MB download; Test 91.7%% "
                             "instead of 94.7%%, and no closest-thing hint for photos turned away).")
    parser.add_argument("--report", type=Path,
                        help="Also write one HTML report (photos embedded, opens offline): counts, a table by vegetable, "
                             "filters and one card per photo with its heatmap.")
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
        "condition_marginal_prediction", "condition_probability", "unknown_score", "clip_unknown_score",
        "rot_evidence", "heatmap", "error",
    ] + probability_columns
    if not calibrated:
        print("Note: this checkpoint has no calibration; run src/calibrate.py to get calibrated probabilities.",
              file=sys.stderr)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rot_scale = checkpoint.get("rot_map", {}).get("scale")
    if args.heatmap:
        args.heatmap.mkdir(parents=True, exist_ok=True)
    clip = None
    if "clip_head" in checkpoint and not args.no_clip:
        from clip_model import ClipModel
        try:
            clip = ClipModel(args.device, **checkpoint["clip_head"]["clip"], species=SPECIES_NAMES[:len(class_names) // 2])
        except ImportError:
            raise SystemExit("This model also uses CLIP: python -m pip install open_clip_torch "
                             "(or add --no-clip to use the vegetable model alone)") from None
    entries = []
    success = 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for path in photos:
            row = {"image": str(path)}
            try:
                photo = open_image(path)
                with torch.inference_mode():
                    logits, rot_maps = forward_with_rot_maps(model, checkpoint, transform(photo).unsqueeze(0).to(device))
                logits, rot_maps = logits.cpu(), rot_maps[0].cpu()
                clip_logits = None
                if clip is not None:
                    features = clip.image_features([photo]).cpu()
                    clip_logits = clip_head_logits(features, checkpoint)
                    probabilities = combined_probabilities(combined_logits(logits, clip_logits, checkpoint), checkpoint)
                    unknown = bool(is_unknown_combined(logits, clip_logits, checkpoint)[0])
                else:
                    probabilities = calibrated_probabilities(logits, checkpoint)
                    unknown = bool(is_unknown(logits, checkpoint)[0])
                index = int(probabilities[0].argmax())
                rot_map = rot_maps[index // len(CONDITION_NAMES)]
                scale = rot_scale or float(rot_map.max()) or 1.0
                species_probs, condition_probs = marginal_probabilities(probabilities)
                species_index = int(species_probs[0].argmax())
                condition_index = int(condition_probs[0].argmax())
                retake = unknown or float(probabilities[0, index]) < args.retake_below
                no_map = unknown or condition_of(index) == "not_judged"  # no rot map where the condition is not judged
                message = UNKNOWN_MESSAGE if unknown else ""
                closest = ""
                if unknown and clip is not None:
                    closest, sentence = clip.message(features[0])
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
                    "condition": "" if unknown else condition_of(index),
                    "probability": round(float(probabilities[0, index]), 4),
                    "uncertain": "yes" if float(probabilities[0, index]) < args.uncertain_below else "no",
                    "species_marginal_prediction": SPECIES_NAMES[species_index],
                    "species_probability": round(float(species_probs[0, species_index]), 4),
                    "condition_marginal_prediction": CONDITION_NAMES[condition_index],
                    "condition_probability": round(float(condition_probs[0, condition_index]), 4),
                    "unknown_score": round(float(unknown_score(logits)[0]), 4),
                    "clip_unknown_score": "" if clip_logits is None else round(float(clip_unknown_score(clip_logits, checkpoint)[0]), 4),
                    "rot_evidence": "" if no_map else round(float(rot_map.max()) / scale, 3),
                    "heatmap": "",
                    "error": "",
                })
                if args.heatmap and not no_map:
                    heatmap_path = args.heatmap / f"{path.stem}_heatmap.jpg"
                    rot_map_overlay(photo, rot_map, scale).save(heatmap_path, quality=90)
                    row["heatmap"] = str(heatmap_path)
                row.update({column: round(float(probabilities[0, i]), 4) for i, column in enumerate(probability_columns)})
                if args.report:
                    entries.append({
                        "name": path.name, "verdict": "unknown" if unknown else condition_of(index),
                        "species": KOREAN[SPECIES_NAMES[index // 2]], "condition": KOREAN[condition_of(index)],
                        "probability": float(probabilities[0, index]), "retake": retake, "retake_message": RETAKE_MESSAGE,
                        "closest": closest, "hint": message[len(UNKNOWN_MESSAGE):].strip(" ()") if unknown else "",
                        "rot": None if no_map else float(row["rot_evidence"]), "photo_uri": image_data_uri(photo),
                        "heat_uri": "" if no_map else image_data_uri(
                            rot_map_overlay(photo, rot_map, scale, size=360, side_by_side=False)),
                    })
                success += 1
                if unknown:
                    print(f"{path.name}: {message}")
                else:
                    flag = " [uncertain]" if row["uncertain"] == "yes" else ""
                    word = "probability" if calibrated else "uncalibrated score"
                    advice = f" - {RETAKE_MESSAGE}" if retake else ""
                    print(f"{path.name}: {class_names[index]} ({KOREAN[SPECIES_NAMES[index // 2]]} "
                          f"{KOREAN[condition_of(index)]}, {word} {100 * row['probability']:.1f}%){flag}{advice}")
            except (OSError, ValueError, RuntimeError) as error:
                row["error"] = f"{type(error).__name__}: {error}"
                if args.report:
                    entries.append({"name": path.name, "verdict": "error", "error": row["error"]})
                print(f"Could not classify {path}: {error}", file=sys.stderr)
            writer.writerow(row)
    print(f"Saved {success}/{len(photos)} successful predictions to {args.output.resolve()}")
    if args.report:
        write_report(args.report, entries, {
            "model": "채소 모델 + CLIP 분류기 (Test 94.7%)" if clip is not None else "채소 모델만 (Test 91.7%)",
            "source": f"입력 폴더: {args.input.expanduser().resolve().name}",
            "notes": [
                "확률은 Valid 사진으로 보정한 값입니다. 이 판정의 확률이 90% 미만이면 다시 찍기를 권합니다.",
                "상한 흔적은 모델이 '비정상' 쪽으로 본 정도입니다. 1이면 Valid 비정상 사진의 중간 수준입니다.",
                "'상한 곳 보기'의 빨간 부분은 모델이 상했다고 본 곳입니다. 7×7칸 단위라 대략적인 위치입니다.",
                "'학습된 데이터가 아닙니다'는 오이·감자·토마토·피망·당근 사진이 아닌 것 같다는 뜻입니다.",
                "인터넷 사진으로 학습해서, 직접 찍은 사진에서는 정확도가 다를 수 있습니다.",
            ],
        })
        print(f"Saved the report to {args.report.resolve()}")
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
