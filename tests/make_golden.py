"""Write tests/golden.json: the answers of the adopted model (model.pth) for one Test photo per class, with and without CLIP.
Run again only when model.pth is replaced on purpose (a new TV version).

    python tests/make_golden.py
"""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def photos() -> list[str]:
    seen, out = set(), []
    for r in csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8")):
        if r["split"] == "test" and r["class_name"] not in seen:
            seen.add(r["class_name"]); out.append(r["image_path"])
    return out


def run(paths: list[str], clip: bool) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp) / "in"; folder.mkdir()
        for p in paths:
            (folder / Path(p).name).symlink_to(REPO / "dataset" / p)
        out = Path(tmp) / "out.csv"
        subprocess.run([sys.executable, str(REPO / "src/predict.py"), "--input", str(folder), "--model", str(REPO / "model.pth"),
                        "--output", str(out), "--threads", "2"] + ([] if clip else ["--no-clip"]), check=True, capture_output=True)
        return {Path(r["image"]).name: {"predicted_class": r["predicted_class"], "probability": float(r["probability"])}
                for r in csv.DictReader(open(out, encoding="utf-8"))}


if __name__ == "__main__":
    paths = photos()
    golden = {"photos": paths, "no_clip": run(paths, False), "full": run(paths, True)}
    json.dump(golden, open(REPO / "tests/golden.json", "w"), indent=1)
    print(json.dumps(golden, indent=1))
