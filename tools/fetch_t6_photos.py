"""Download the photos of the sealed blind test set T6 again (results/t6_test/t6_test.csv) after the working folder was lost.

    python tools/fetch_t6_photos.py $DEPP_WORK/t6test/img

Saves <id>.jpg per row. iNaturalist: the open-data bucket; Openverse: the image URL of the Openverse record; Commons: the file
page; dataset photos: the same file of the same public dataset (Hugging Face parquet row, Mendeley / Zenodo zip member).
Checks the seal first. Prints rows that could not be fetched; nothing is guessed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "depp-env22-student-project (university coursework)"}


def get(url: str, tries: int = 5) -> bytes | None:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=300) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(int(e.headers.get("retry-after") or 0) or 20 * (k + 1))
        except Exception:
            time.sleep(20 * (k + 1))
    return None


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((REPO / "results/t6_test/t6_test.csv").read_bytes()).hexdigest()
    sealed = re.findall(r"SHA-256[^`]*`([0-9a-f]{64})`", (REPO / "results/t6_test/RULES.md").read_text(encoding="utf-8"))
    assert sealed and sealed[-1] == digest, "t6_test.csv does not match its seal"
    rows = [r for r in csv.DictReader(open(REPO / "results/t6_test/t6_test.csv", encoding="utf-8")) if not (out / f"{r['id']}.jpg").exists()]
    missing = []
    datasets = {}
    for r in rows:
        i = r["id"]; data = None
        if i.startswith("inat-"):
            for ext in ("jpg", "jpeg", "png"):
                data = get(f"https://inaturalist-open-data.s3.amazonaws.com/photos/{i[5:]}/medium.{ext}")
                if data:
                    break
        elif i.startswith("commons-"):
            title = urllib.parse.unquote(r["page"].rsplit("/", 1)[-1]).replace("File:", "")
            data = get(f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(title)}?width=800"); time.sleep(5)
        elif r["source"] == "openverse":
            meta = get(f"https://api.openverse.org/v1/images/{i}/")
            url = json.loads(meta).get("url") if meta else None
            data = get(url) if url else None; time.sleep(3)
        elif r["source"] == "dataset":
            datasets.setdefault(i.split("-")[0], []).append(r); continue
        if data:
            (out / f"{i}.jpg").write_bytes(data)
        else:
            missing.append(i)
    if "khatun" in datasets:
        import pyarrow.parquet as pq
        p = out / "_khatun.parquet"
        p.write_bytes(get("https://huggingface.co/datasets/Project-AgML/tomato_quality_classification/resolve/main/raw/train-00000-of-00001.parquet"))
        imgs = pq.read_table(p, columns=["image"]).column("image").to_pylist()
        for r in datasets["khatun"]:
            (out / f"{r['id']}.jpg").write_bytes(imgs[int(r["id"].split("-")[1])]["bytes"])
        p.unlink()
    if "mtom" in datasets:
        z = zipfile.ZipFile(io.BytesIO(get("https://data.mendeley.com/public-files/datasets/vy5y8p2kdb/files/0bcb26f1-6dbd-478c-9166-95b9ce9ccbb2/file_downloaded")))
        for r in datasets["mtom"]:
            _, split, stem = r["id"].split("-", 2)
            hit = [n for n in z.namelist() if f"/{split}/images/" in n and n.split("/")[-1].split("_jpg")[0] == stem]
            if hit:
                (out / f"{r['id']}.jpg").write_bytes(z.read(hit[0]))
            else:
                missing.append(r["id"])
    if "malay" in datasets:
        z = zipfile.ZipFile(io.BytesIO(get("https://zenodo.org/records/20843858/files/Malaysian%20Food%20Freshness%20Dataset%20for%20Edge%20AI%20Classification.zip?download=1")))
        for r in datasets["malay"]:
            _, kind, stem = r["id"].split("-", 2)
            hit = [n for n in z.namelist() if n.split("/")[-1].rsplit(".", 1)[0] == f"{'Rotten' if kind.startswith('Rotten') else 'Fresh'}_{kind}_{stem}"]
            if hit:
                (out / f"{r['id']}.jpg").write_bytes(z.read(hit[0]))
            else:
                missing.append(r["id"])
    have = sum((out / f"{r['id']}.jpg").exists() for r in csv.DictReader(open(REPO / "results/t6_test/t6_test.csv", encoding="utf-8")))
    print("T6 photos on disk:", have, "of 175; could not fetch:", missing)


if __name__ == "__main__":
    main()
