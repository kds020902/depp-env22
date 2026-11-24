"""Write tests/golden.json for one Test photo per class:
- "full": the system's answers (predict.py; needs CLIP);
- "vegetable_model_output": what the vegetable model, one part of the system, puts out for the same photos (its most likely
  class and calibrated probability). It is not an answer of the system; the fast tests use it to see that the weights and
  the image pipeline are unchanged where CLIP is not installed.
Run again only when model.pth is replaced on purpose (a new TV version).

    python tests/make_golden.py
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from calibration import calibrated_probabilities  # noqa: E402
from images import make_transform, read_image  # noqa: E402
from labels import CLASS_NAMES  # noqa: E402
from networks import input_size, load_model  # noqa: E402


def photos() -> list[str]:
    seen, out = set(), []
    for r in csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")):
        if r["split"] == "test" and r["class_name"] not in seen:
            seen.add(r["class_name"]); out.append(r["image_path"])
    return out


def system_answers(paths: list[str]) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "in"; folder.mkdir()
        for p in paths:
            (folder / Path(p).name).symlink_to(REPO / "dataset" / p)
        out = Path(tmp) / "out.csv"
        subprocess.run([sys.executable, str(REPO / "src/predict.py"), "--input", str(folder), "--model", str(REPO / "model.pth"),
                        "--output", str(out), "--threads", "2"], check=True, capture_output=True)
        return {Path(r["image"]).name: {"predicted_class": r["predicted_class"], "probability": float(r["probability"])}
                for r in csv.DictReader(open(out, encoding="utf-8"))}


def vegetable_model_output(paths: list[str]) -> dict:
    model, checkpoint = load_model(REPO / "model.pth", torch.device("cpu"))
    transform = make_transform(input_size(checkpoint))
    with torch.inference_mode():
        p = calibrated_probabilities(model(torch.stack([read_image(REPO / "dataset" / path, transform) for path in paths])), checkpoint)
    return {Path(path).name: {"predicted_class": CLASS_NAMES[int(row.argmax())], "probability": round(float(row.max()), 4)}
            for path, row in zip(paths, p)}


if __name__ == "__main__":
    paths = photos()
    golden = {"photos": paths, "vegetable_model_output": vegetable_model_output(paths), "full": system_answers(paths)}
    json.dump(golden, open(REPO / "tests/golden.json", "w"), indent=1)
    print(json.dumps(golden, indent=1))
