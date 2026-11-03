"""Download again the extra training photos of the CLIP head (experiments/not_five_class/added_photos.csv), after the working
folder was lost. Resumable: finished sources are skipped. Writes OUT/added_photos_local.csv (row number, local path or empty).

    python tools/rebuild_head_photos.py $DEPP_WORK/head

Kaggle datasets are downloaded through the public API (no account needed for these), one at a time; only the listed files are
extracted and the zip is deleted. Rows whose source could not be found are reported, not guessed.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "depp-env22-student-project (university coursework)"}
KAGGLE = {"kritikseth": "kritikseth/fruit-and-vegetable-image-recognition", "aelchimminut": "aelchimminut/fruits262",
          "jimpax": "jimpax/datasetv13", "vetion": "vetion/vege-quality", "sujaykapadnis": "sujaykapadnis/cucumber-disease-recognition-dataset",
          "mdatikurrahman3111": "mdatikurrahman3111/vegetables-quality-dataset-2", "razwansultan": "razwansultan/tomato-disease-dataset-both-leaf-and-fruit",
          "user2036": "user2036/vetable-dataset-without-duplicacy"}


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
        except Exception as e:
            print("retry", url[:80], e, flush=True); time.sleep(20 * (k + 1))
    return None


def match(names: list[str], ref: str) -> str | None:
    """The zip member for a listed path: the listed path without its first folders, matched at the end. "name.*" matches any
    extension; a trailing " (crop ...)" is ignored here."""
    ref = ref.split(" (crop")[0]
    parts = ref.split("/")
    any_ext = parts[-1].endswith(".*")
    strip = lambda n: n.rsplit(".", 1)[0] if any_ext else n
    for cut in range(0, len(parts)):
        tail = "/".join(parts[cut:]); tail = tail[:-2] if any_ext else tail
        hits = [n for n in names if strip(n).endswith("/" + tail) or strip(n) == tail]
        if len(hits) == 1:
            return hits[0]
        if hits:
            return sorted(hits, key=len)[0]
    return None


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(REPO / "experiments/not_five_class/added_photos.csv", encoding="utf-8")))
    state_path = out / "local.json"
    local = json.load(open(state_path)) if state_path.exists() else {}
    save = lambda: json.dump(local, open(state_path, "w"))
    # Kaggle datasets, one zip at a time
    for key, ref in KAGGLE.items():
        todo = [i for i, r in enumerate(rows) if f"kaggle.com/datasets/{ref}" in r["source"] and not local.get(str(i))]
        if not todo:
            continue
        zpath = out / (ref.replace("/", "__") + ".zip")
        print("kaggle", ref, len(todo), flush=True)
        if get(f"https://www.kaggle.com/api/v1/datasets/download/{ref}", zpath) is None:
            continue
        with zipfile.ZipFile(zpath) as z:
            names = [n for n in z.namelist() if not n.endswith("/")]
            for i in todo:
                m = match(names, rows[i]["reference"])
                if m is None:
                    local[str(i)] = ""; continue
                dst = out / "kaggle" / key / m
                dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(z.read(m)); local[str(i)] = str(dst)
        zpath.unlink(); save()
        print("  found", sum(bool(local[str(i)]) for i in todo), "of", len(todo), flush=True)
    # other archives: (name, url, which rows)
    archives = [
        ("caltech101", "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1", lambda r: r["source"].startswith("Caltech-101")),
        ("zenodo_cucumber", "https://zenodo.org/api/records/10081197/files/Cucumber%20Dataset.zip/content", lambda r: "zenodo.org/records/10081197" in r["source"]),
        ("mendeley_cucumber", "https://data.mendeley.com/public-api/zip/zncrtr2yhn/download/1", lambda r: r["source"] == "" and r["reference"].startswith("CUCUMBER/")),
        ("zenodo_potato", "https://zenodo.org/api/records/20616991/files/dataset_papas_hibrido.zip/content",
         lambda r: r["source"] == "" and r["reference"].split("/")[0] in ("train", "val", "test")),
    ]
    for name, url, which in archives:
        todo = [i for i, r in enumerate(rows) if which(r) and not local.get(str(i))]
        if not todo:
            continue
        zpath = out / f"{name}.zip"
        print("archive", name, len(todo), flush=True)
        if get(url, zpath) is None:
            continue
        z = zipfile.ZipFile(zpath); names = [n for n in z.namelist() if not n.endswith("/")]
        inner = [n for n in names if n.endswith(".tar.gz")]
        if inner:   # Caltech-101: the photos are in a tar.gz inside the zip
            import tarfile
            tf = tarfile.open(fileobj=io.BytesIO(z.read(inner[0]))); members = {m.name: m for m in tf.getmembers() if m.isfile()}
            read, names = (lambda n: tf.extractfile(members[n]).read()), list(members)
        else:
            read = z.read
        for i in todo:
            ref = rows[i]["reference"]; m = match(names, ref)
            if m is None:
                local[str(i)] = ""; continue
            dst = out / name / (m.replace("/", "__") if " (crop" not in ref else m.replace("/", "__") + ".crop.jpg")
            dst.parent.mkdir(parents=True, exist_ok=True); data = read(m)
            if " (crop" in ref:
                from PIL import Image, ImageOps
                box = [int(v) for v in ref.split(" (crop ")[1].rstrip(")").split()]
                ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB").crop(box).save(dst, quality=95)
            else:
                dst.write_bytes(data)
            local[str(i)] = str(dst)
        z.close(); zpath.unlink(); save()
        print("  found", sum(bool(local[str(i)]) for i in todo), "of", len(todo), flush=True)
    # Hugging Face Places365 sample: parquet rows
    todo = [i for i, r in enumerate(rows) if "places365-mini-sample-easy" in r["source"] and not local.get(str(i))]
    if todo:
        import pyarrow.parquet as pq
        ppath = out / "places_test.parquet"
        get("https://huggingface.co/datasets/dpdl-benchmark/places365-mini-sample-easy/resolve/main/data/test-00000-of-00001.parquet", ppath)
        imgs = pq.read_table(ppath).column("image").to_pylist()
        for i in todo:
            k = int(rows[i]["reference"].split(" row ")[1].split()[0])
            dst = out / "places" / f"row{k}.jpg"; dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(imgs[k]["bytes"])
            local[str(i)] = str(dst)
        save()
    # Openverse rows: the image URL from the Openverse record
    for i, r in enumerate(rows):
        if local.get(str(i)) or not r["reference"].startswith("openverse "):
            continue
        oid = r["reference"].split()[1]
        meta = get(f"https://api.openverse.org/v1/images/{oid}/")
        url = json.loads(meta).get("url") if meta else None
        dst = out / "openverse" / f"{oid}.jpg"; dst.parent.mkdir(parents=True, exist_ok=True)
        local[str(i)] = str(dst) if url and get(url, dst) is not None else ""
        save(); time.sleep(3)
    save()
    found = sum(bool(v) for v in local.values())
    print("rows", len(rows), "handled", len(local), "found", found, flush=True)


if __name__ == "__main__":
    main()
