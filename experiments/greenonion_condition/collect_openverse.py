"""Candidate photos for spoiled green onions from Openverse (openly licensed), for labelling by eye. No model picks photos.

    python experiments/greenonion_condition/collect_openverse.py OUT_DIR

Writes OUT_DIR/manifest.csv (file, title, creator, license, source_url, image_url, query) and the thumbnails.
Photos stay outside the repository.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

UA = {"User-Agent": "veg-dataset-research/0.1 (student project)"}
QUERIES = ["wilted green onions", "yellowed green onions", "rotten spring onion", "old scallions", "wilted scallions",
           "spoiled leek", "rotting leeks", "slimy scallions", "moldy green onion", "green onion compost"]


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for q in QUERIES:
        url = "https://api.openverse.org/v1/images/?" + urlencode({"q": q, "page_size": 20})
        try:
            with urlopen(Request(url, headers=UA), timeout=30) as r:
                results = json.load(r).get("results", [])
        except Exception as e:
            print("query failed", q, e); continue
        for item in results:
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            thumb = item.get("thumbnail") or item.get("url")
            name = f"ov_{len(rows):03d}.jpg"
            try:
                with urlopen(Request(thumb, headers=UA), timeout=30) as r:
                    (out / name).write_bytes(r.read())
            except Exception:
                continue
            rows.append({"file": name, "title": item.get("title", ""), "creator": item.get("creator", ""),
                         "license": f"{item.get('license', '')} {item.get('license_version', '')}".strip(),
                         "source_url": item.get("foreign_landing_url", ""), "image_url": item.get("url", ""), "query": q})
            time.sleep(0.3)
        time.sleep(1)
    with open(out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} candidates")


if __name__ == "__main__":
    main()
