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
    """The zip member for a listed path: the listed path without its first folder (the local folder name), matched at the end."""
    parts = ref.split("/")
    for cut in range(1, len(parts)):
        tail = "/" + "/".join(parts[cut:])
        hits = [n for n in names if n.endswith(tail) or n == tail[1:]]
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
        todo = [i for i, r in enumerate(rows) if f"kaggle.com/datasets/{ref}" in r["source"] and str(i) not in local]
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
    # Openverse rows: the image URL from the Openverse record
    for i, r in enumerate(rows):
        if str(i) in local or not r["reference"].startswith("openverse "):
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
