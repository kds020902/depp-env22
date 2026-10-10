"""Cut the Zenodo cucumbers used in this experiment out of the original photos.

    python experiments/asian_cucumber/crop_zenodo.py "Cucumber Dataset.zip" out_dir [added_photos_J2.csv]

The zip is https://zenodo.org/records/10081197 (CC BY 4.0). The optional third argument picks the photo list: the default
added_photos.csv holds the 60 of candidate J, added_photos_J2.csv the 140 of candidate J2. Each photo shows one cucumber on
black cloth under a ruler and three colour cards. crop_box_xyxy is the box around the fruit in original pixels (fruit + 15%
margin, widened to at most 2:1); the crop is saved with the long side at 640 px.
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
    listing = HERE / (sys.argv[3] if len(sys.argv) > 3 else "added_photos.csv")
    with listing.open(encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle) if r["crop_box_xyxy"]]
    for r in rows:
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(archive.read(r["reference"])))).convert("RGB")
        crop = image.crop(tuple(int(v) for v in r["crop_box_xyxy"].split()))
        crop.thumbnail((640, 640))
        crop.save(out / (Path(r["reference"]).stem.replace(" ", "_") + ".jpg"), quality=92)
    print(f"saved {len(rows)} crops to {out}")


if __name__ == "__main__":
    main()
