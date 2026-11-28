"""Download a list of candidate photo URLs (e.g. found by agy) for review by eye. No model picks photos.

    python experiments/greenonion_condition/fetch_list.py LIST.csv OUT_DIR

LIST.csv columns: image_url, source_page, license, note (extra columns are kept). Writes OUT_DIR/manifest.csv with
file, sha256, width, height, status for every row; failed downloads are recorded, not retried. Photos stay outside the repo.
"""

from __future__ import annotations

import csv
import hashlib
import io
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from PIL import Image

UA = {"User-Agent": "veg-dataset-research/0.1 (student project)"}
LIMIT = 30 * 1024 * 1024


def safe_url(url: str) -> str:
    p = urlsplit(url.strip())
    if p.scheme not in ("http", "https"):
        raise ValueError("not an http(s) URL")
    return urlunsplit((p.scheme, p.netloc, quote(p.path, safe="/%:@"), quote(p.query, safe="=&%:@/"), ""))


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(src, encoding="utf-8-sig", newline="")))
    done = []
    for i, row in enumerate(rows):
        rec = dict(row, file="", sha256="", width="", height="", status="")
        try:
            with urlopen(Request(safe_url(row["image_url"]), headers=UA), timeout=40) as r:
                data = r.read(LIMIT + 1)
            if len(data) > LIMIT:
                raise ValueError("too large")
            with Image.open(io.BytesIO(data)) as im:
                im.verify()
            with Image.open(io.BytesIO(data)) as im:
                rec["width"], rec["height"] = im.size
            name = f"go_{i:03d}.jpg"
            (out / name).write_bytes(data)
            rec.update(file=name, sha256=hashlib.sha256(data).hexdigest(), status="ok")
        except Exception as e:
            rec["status"] = f"failed: {type(e).__name__}: {e}"[:200]
        done.append(rec)
        time.sleep(0.5)
    with open(out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(done[0]))
        w.writeheader()
        w.writerows(done)
    ok = sum(r["status"] == "ok" for r in done)
    print(f"{ok} of {len(done)} downloaded")


if __name__ == "__main__":
    main()
