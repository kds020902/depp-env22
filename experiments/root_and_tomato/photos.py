"""Put the photos of new_train.csv in the working folder: $DEPP_WORK/rt/img/<id>.jpg (re-runnable, skips photos on disk).

    python experiments/root_and_tomato/photos.py

- Kaggle rows: the public download API (no account needed for these datasets), one zip at a time; the listed member is read,
  rotated by its EXIF tag, cropped when the reference ends with " (crop x0 y0 x1 y1)", and saved. The zip is kept in
  $DEPP_WORK/rt/zips so a second run does not download it again.
- iNaturalist rows ("inat-<photo id>"): the public open-data bucket, 'large' size.
Rows that cannot be found are printed, never guessed.
"""

from __future__ import annotations

import csv
import io
import os
import time
import urllib.request
import zipfile
from pathlib import Path

from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
S = Path(os.environ["DEPP_WORK"])
OUT, ZIPS = S / "rt/img", S / "rt/zips"
UA = {"User-Agent": "depp-env22-student-project (university coursework)"}


def get(url: str, dst: Path | None = None, tries: int = 4) -> bytes | None:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
                if dst is None:
                    return r.read()
                with open(dst, "wb") as f:
                    while chunk := r.read(1 << 22):
                        f.write(chunk)
                return b""
        except Exception as e:  # noqa: BLE001
            print("retry", url[:80], e, flush=True); time.sleep(20 * (k + 1))
    return None


def save(data: bytes, ref: str, dst: Path):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    if " (crop " in ref:
        image = image.crop(tuple(int(v) for v in ref.split(" (crop ")[1].rstrip(")").split()))
    image.thumbnail((1024, 1024))
    image.save(dst, quality=92)


def main():
    OUT.mkdir(parents=True, exist_ok=True); ZIPS.mkdir(parents=True, exist_ok=True)
    rows = [r for r in csv.DictReader(open(HERE / "new_train.csv", encoding="utf-8")) if not (OUT / f"{r['id']}.jpg").exists()]
    missing = []
    for ref in sorted({r["source"].split("/datasets/")[1] for r in rows if "kaggle.com/datasets/" in r["source"]}):
        z = ZIPS / (ref.replace("/", "__") + ".zip")
        if not z.exists() and get(f"https://www.kaggle.com/api/v1/datasets/download/{ref}", z) is None:
            missing += [r["id"] for r in rows if ref in r["source"]]; continue
        with zipfile.ZipFile(z) as archive:
            names = set(archive.namelist())
            for r in rows:
                if ref not in r["source"]:
                    continue
                member = r["reference"].split(" (crop ")[0]
                if member in names:
                    save(archive.read(member), r["reference"], OUT / f"{r['id']}.jpg")
                else:
                    missing.append(r["id"])
    for r in rows:
        if r["id"].startswith("inat-"):
            pid = r["id"][5:]
            data = next((d for ext in ("jpg", "jpeg", "png")
                         if (d := get(f"https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/large.{ext}", tries=2))), None)
            if data:
                save(data, r["reference"], OUT / f"{r['id']}.jpg")
            else:
                missing.append(r["id"])
    all_rows = list(csv.DictReader(open(HERE / "new_train.csv", encoding="utf-8")))
    print("photos on disk:", sum((OUT / f"{r['id']}.jpg").exists() for r in all_rows), "of", len(all_rows), "| not found:", missing)


if __name__ == "__main__":
    main()
