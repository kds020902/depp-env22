"""Rebuild the 192 real-scene vegetable crops in lvis_vegetables.csv from the LVIS fruits-and-vegetables Kaggle zip.

    python experiments/real_scene_threshold/crop_lvis.py lvis-fruits-and-vegetables-dataset.zip lvis_vegetables
    python src/calibrate.py --extra-photos lvis_vegetables --extra-name "LVIS real-scene vegetable crops"

Each crop is the YOLO box widened by 10% on every side, rotated by its EXIF tag, and shrunk to at most 640 pixels.
Dataset: https://www.kaggle.com/datasets/henningheyen/lvis-fruits-and-vegetables-dataset (MIT)
"""

import csv
import io
import sys
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent


def main():
    archive, out = zipfile.ZipFile(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    with (HERE / "lvis_vegetables.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(archive.read(row["lvis_image"])))).convert("RGB")
        cx, cy, w, h = map(float, row["yolo_box_cx_cy_w_h"].split())
        width, height = image.size
        bw, bh = w * width, h * height
        box = ((cx - w / 2) * width - 0.1 * bw, (cy - h / 2) * height - 0.1 * bh,
               (cx + w / 2) * width + 0.1 * bw, (cy + h / 2) * height + 0.1 * bh)
        crop = image.crop((max(0, int(box[0])), max(0, int(box[1])), min(width, int(box[2])), min(height, int(box[3]))))
        crop.thumbnail((640, 640))
        crop.save(out / row["crop_name"], quality=92)
    print(f"{len(rows)} crops in {out}")


if __name__ == "__main__":
    main()
