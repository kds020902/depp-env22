"""Classify a photo or a directory with the adopted system (vegetable kind x fresh/rotten).

The vegetable model and the CLIP classifiers are combined by decision.decide(): a photo is turned away when it does not look
like one of the vegetables, and the second CLIP head may answer onion, sweet potato or green onion. CLIP needs
`pip install open_clip_torch` and downloads about 600 MB on first use; --no-clip uses the vegetable model alone (five
vegetables only), --five-only skips the onion / sweet potato / green onion head.

    python src/predict.py --input my_photos --model model.pth [--report report.html] [--heatmap heatmaps]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch

from calibration import marginal_probabilities, unknown_score
from decision import clip_head_logits, clip_unknown_score, decide
from images import IMAGE_EXTENSIONS, make_transform, open_image
from labels import CONDITION_NAMES, KOREAN, SPECIES_NAMES, UNKNOWN_MESSAGE, condition_of
from networks import forward_with_rot_maps, input_size, load_model, resolve_device, rot_map_overlay
from report import image_data_uri, write_report

RETAKE_MESSAGE = "확신이 낮습니다. 채소 하나만, 밝은 곳에서, 상한 부분이 보이게 가까이 다시 찍어 보세요."
FIELDS = ["image", "known", "message", "closest_guess", "retake", "predicted_class", "species", "condition", "probability",
          "uncertain", "species_marginal_prediction", "species_probability", "condition_marginal_prediction",
          "condition_probability", "unknown_score", "clip_unknown_score", "rot_evidence", "heatmap", "error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
                        help="Use the vegetable model alone, without CLIP (faster, no 600 MB download; five vegetables only, "
                             "and no closest-thing hint for photos turned away).")
    parser.add_argument("--five-only", action="store_true",
                        help="Do not use the onion / sweet potato / green onion head (answers are one of the five vegetables).")
    parser.add_argument("--report", type=Path,
                        help="Also write one HTML report (photos embedded, opens offline): counts, a table by vegetable, "
                             "filters and one card per photo with its heatmap.")
    parser.add_argument("--heatmap", type=Path,
                        help="Folder for heatmaps: each photo next to a copy where the parts the model saw as rotten "
                             "are coloured red (stronger red = stronger evidence).")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


def find_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


class Classifier:
    """The adopted system, one photo at a time."""

    def __init__(self, model_path: Path, device: torch.device, use_clip: bool = True, new_vegetables: bool = True):
        self.model, self.checkpoint = load_model(model_path, device)
        self.device = device
        self.transform = make_transform(input_size(self.checkpoint))
        self.calibrated = "calibration" in self.checkpoint
        self.rot_scale = self.checkpoint.get("rot_map", {}).get("scale")
        self.clip = None
        if use_clip and "clip_head" in self.checkpoint:
            from clip_model import ClipModel
            n_kinds = len(self.checkpoint["new_vegetable_head"]["class_names"] if new_vegetables and "new_vegetable_head" in self.checkpoint
                          else self.checkpoint["class_names"]) // len(CONDITION_NAMES)
            try:   # the closest-thing hint names the kinds this system answers
                self.clip = ClipModel(device.type, **self.checkpoint["clip_head"]["clip"], species=SPECIES_NAMES[:n_kinds])
            except ImportError:
                raise SystemExit("This model also uses CLIP: python -m pip install open_clip_torch "
                                 "(or add --no-clip to use the vegetable model alone)") from None
        self.new_vegetables = new_vegetables and self.clip is not None and "new_vegetable_head" in self.checkpoint
        self.class_names = self.checkpoint["new_vegetable_head"]["class_names"] if self.new_vegetables else self.checkpoint["class_names"]

    def describe(self) -> str:
        if self.new_vegetables:
            return "채소 모델 + CLIP 분류기 2개 (8종: 다섯 채소 + 양파·고구마·대파)"
        return "채소 모델 + CLIP 분류기 (다섯 채소)" if self.clip is not None else "채소 모델만 (다섯 채소)"

    def classify(self, photo, retake_below: float) -> dict:
        """Everything predict.py shows for one photo."""
        with torch.inference_mode():
            logits, rot_maps = forward_with_rot_maps(self.model, self.checkpoint, self.transform(photo).unsqueeze(0).to(self.device))
        logits, rot_maps = logits.cpu(), rot_maps[0].cpu()
        features = self.clip.image_features([photo]).cpu() if self.clip is not None else None
        a = decide(logits, self.checkpoint, features, self.new_vegetables)
        index, unknown, probability = int(a.index[0]), bool(a.unknown[0]), float(a.probability[0])
        species_probs, condition_probs = marginal_probabilities(a.probabilities)
        # the vegetable model's rot maps cover its own (five) vegetables only
        kind = index // len(CONDITION_NAMES)
        rot_map = rot_maps[min(kind, rot_maps.shape[0] - 1)]
        no_map = unknown or kind >= rot_maps.shape[0] or condition_of(index) == "not_judged"
        scale = self.rot_scale or float(rot_map.max()) or 1.0
        retake = unknown or probability < retake_below
        message, closest = (UNKNOWN_MESSAGE if unknown else ""), ""
        if unknown and self.clip is not None:
            closest, sentence = self.clip.message(features[0])
            message = f"{UNKNOWN_MESSAGE} ({sentence})"
        elif retake:
            message = RETAKE_MESSAGE
        return {
            "index": index, "unknown": unknown, "probability": probability, "probabilities": a.probabilities[0],
            "species_index": int(species_probs[0].argmax()), "species_probability": float(species_probs[0].max()),
            "condition_index": int(condition_probs[0].argmax()), "condition_probability": float(condition_probs[0].max()),
            "retake": retake, "message": message, "closest": closest,
            "unknown_score": float(unknown_score(logits)[0]),
            "clip_unknown_score": None if features is None else
            float(clip_unknown_score(clip_head_logits(features, self.checkpoint), self.checkpoint, features)[0]),
            "rot_map": None if no_map else rot_map, "rot_evidence": None if no_map else float(rot_map.max()) / scale, "scale": scale,
        }


def csv_row(path: Path, r: dict, names: list[str], uncertain_below: float) -> dict:
    index, unknown = r["index"], r["unknown"]
    row = {
        "image": str(path), "known": "no" if unknown else "yes", "message": r["message"], "closest_guess": r["closest"],
        "retake": "yes" if r["retake"] else "no", "predicted_class": "unknown" if unknown else names[index],
        "species": "" if unknown else SPECIES_NAMES[index // 2], "condition": "" if unknown else condition_of(index),
        "probability": round(r["probability"], 4), "uncertain": "yes" if r["probability"] < uncertain_below else "no",
        "species_marginal_prediction": SPECIES_NAMES[r["species_index"]], "species_probability": round(r["species_probability"], 4),
        "condition_marginal_prediction": CONDITION_NAMES[r["condition_index"]],
        "condition_probability": round(r["condition_probability"], 4), "unknown_score": round(r["unknown_score"], 4),
        "clip_unknown_score": "" if r["clip_unknown_score"] is None else round(r["clip_unknown_score"], 4),
        "rot_evidence": "" if r["rot_evidence"] is None else round(r["rot_evidence"], 3), "heatmap": "", "error": "",
    }
    row.update({f"prob_{name}": round(float(r["probabilities"][i]), 4) for i, name in enumerate(names)})
    return row


def console_line(path: Path, r: dict, names: list[str], uncertain_below: float, calibrated: bool) -> str:
    if r["unknown"]:
        return f"{path.name}: {r['message']}"
    index = r["index"]
    flag = " [uncertain]" if r["probability"] < uncertain_below else ""
    word = "probability" if calibrated else "uncalibrated score"
    advice = f" - {RETAKE_MESSAGE}" if r["retake"] else ""
    return (f"{path.name}: {names[index]} ({KOREAN[SPECIES_NAMES[index // 2]]} {KOREAN[condition_of(index)]}, "
            f"{word} {100 * r['probability']:.1f}%){flag}{advice}")


def report_entry(path: Path, photo, r: dict) -> dict:
    index, unknown = r["index"], r["unknown"]
    return {
        "name": path.name, "verdict": "unknown" if unknown else condition_of(index),
        "species": KOREAN[SPECIES_NAMES[index // 2]], "condition": KOREAN[condition_of(index)],
        "probability": r["probability"], "retake": r["retake"], "retake_message": RETAKE_MESSAGE, "closest": r["closest"],
        "hint": r["message"][len(UNKNOWN_MESSAGE):].strip(" ()") if unknown else "",
        "rot": None if r["rot_evidence"] is None else round(r["rot_evidence"], 3), "photo_uri": image_data_uri(photo),
        "heat_uri": "" if r["rot_map"] is None else image_data_uri(rot_map_overlay(photo, r["rot_map"], r["scale"], size=360, side_by_side=False)),
    }


def write_html(path: Path, entries: list[dict], system: Classifier, source: Path) -> None:
    names = "오이·감자·토마토·피망·당근·양파·고구마·대파" if system.new_vegetables else "오이·감자·토마토·피망·당근"
    notes = [
        "확률은 Valid 사진으로 보정한 값입니다. 이 판정의 확률이 90% 미만이면 다시 찍기를 권합니다.",
        "상한 흔적은 모델이 '비정상' 쪽으로 본 정도입니다. 1이면 Valid 비정상 사진의 중간 수준입니다.",
        "'상한 곳 보기'의 빨간 부분은 모델이 상했다고 본 곳입니다. 7×7칸 단위라 대략적인 위치입니다.",
        f"'학습된 데이터가 아닙니다'는 {names} 사진이 아닌 것 같다는 뜻입니다."
        + (" 대파는 종류만 알아보고 상태는 판정하지 않습니다." if system.new_vegetables else ""),
        "인터넷 사진으로 학습해서, 직접 찍은 사진에서는 정확도가 다를 수 있습니다.",
    ]
    if system.new_vegetables:
        notes.append("'상한 곳 보기'는 다섯 채소에만 있습니다. 양파·고구마·대파는 CLIP 분류기가 답해서 상한 곳 지도가 없습니다.")
    write_report(path, entries, {"model": system.describe(), "source": f"입력 폴더: {source.name}", "notes": notes})


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    system = Classifier(args.model.expanduser().resolve(), resolve_device(args.device),
                        use_clip=not args.no_clip, new_vegetables=not args.five_only)
    source = args.input.expanduser().resolve()
    photos = find_images(source)
    if not photos:
        raise ValueError("No supported image files were found.")
    if not system.calibrated:
        print("Note: this checkpoint has no calibration; run src/calibrate.py to get calibrated probabilities.", file=sys.stderr)
    names = system.class_names
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.heatmap:
        args.heatmap.mkdir(parents=True, exist_ok=True)
    entries, success = [], 0
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS + [f"prob_{name}" for name in names])
        writer.writeheader()
        for path in photos:
            try:
                photo = open_image(path)
                r = system.classify(photo, args.retake_below)
                row = csv_row(path, r, names, args.uncertain_below)
                if args.heatmap and r["rot_map"] is not None:
                    row["heatmap"] = str(args.heatmap / f"{path.stem}_heatmap.jpg")
                    rot_map_overlay(photo, r["rot_map"], r["scale"]).save(row["heatmap"], quality=90)
                if args.report:
                    entries.append(report_entry(path, photo, r))
                print(console_line(path, r, names, args.uncertain_below, system.calibrated))
                success += 1
            except (OSError, ValueError, RuntimeError) as error:
                row = {"image": str(path), "error": f"{type(error).__name__}: {error}"}
                if args.report:
                    entries.append({"name": path.name, "verdict": "error", "error": row["error"]})
                print(f"Could not classify {path}: {error}", file=sys.stderr)
            writer.writerow(row)
    print(f"Saved {success}/{len(photos)} successful predictions to {args.output.resolve()}")
    if args.report:
        write_html(args.report, entries, system, source)
        print(f"Saved the report to {args.report.resolve()}")


if __name__ == "__main__":
    main()
