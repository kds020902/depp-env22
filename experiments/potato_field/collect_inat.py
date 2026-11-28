"""Screen iNaturalist potato (Solanum tuberosum) observation photos for tubers (no model is used to pick photos).

    python experiments/potato_field/collect_inat.py OUT_DIR [N_OBSERVATIONS]

1. Lists observations with CC-licensed photos (newest first), up to 3 photos each.
2. Downloads the "small" size (240 px) for screening and records observation id, observer, license, attribution.
3. Marks photos with little green (leaf / plant photos are mostly green) as `screen_keep` by a plain colour rule:
   share of pixels with G > R + 10 and G > B + 10 below 0.25. Labels are then given by eye on contact sheets.
Group for any later split = "inat:obs:<observation id>" (Codex rule); same observer is recorded too.
Images stay outside the repository.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image

API = "https://api.inaturalist.org/v1/observations"
UA = {"User-Agent": "veg-dataset-research/0.1 (student project)"}
LICENSES = "cc0,cc-by,cc-by-sa,cc-by-nc,cc-by-nc-sa"


def get(url, params=None):
    with urlopen(Request(url + ("?" + urlencode(params) if params else ""), headers=UA), timeout=60) as r:
        return r.read()


def green_share(data):
    a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB")).astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return float(((g > r + 10) & (g > b + 10)).mean())


def main():
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    taxon = json.loads(get("https://api.inaturalist.org/v1/taxa", {"q": "Solanum tuberosum", "rank": "species"}))["results"][0]["id"]
    rows, id_below, seen = [], None, 0
    while seen < want:
        p = {"taxon_id": taxon, "photos": "true", "photo_license": LICENSES, "per_page": 200, "order_by": "id", "order": "desc"}
        if id_below:
            p["id_below"] = id_below
        res = json.loads(get(API, p))["results"]
        if not res:
            break
        for o in res:
            seen += 1
            for k, ph in enumerate(o.get("photos", [])[:3]):
                if not ph.get("license_code"):
                    continue
                url = ph["url"].replace("square", "small")
                try:
                    data = get(url)
                except Exception:
                    continue
                name = f"inat_{o['id']}_{k}.jpg"
                (out / name).write_bytes(data)
                rows.append({"file": name, "observation_id": o["id"], "photo_id": ph["id"], "observer": (o.get("user") or {}).get("login", ""),
                             "license": ph["license_code"], "attribution": ph.get("attribution", ""),
                             "source_url": f"https://www.inaturalist.org/observations/{o['id']}",
                             "image_url_large": ph["url"].replace("square", "large"), "green_share": round(green_share(data), 3),
                             "source_group": f"inat:obs:{o['id']}"})
                time.sleep(0.15)
        id_below = res[-1]["id"]
        time.sleep(1.0)
        print(f"observations {seen}, photos {len(rows)}", flush=True)
    for r in rows:
        r["screen_keep"] = r["green_share"] < 0.25
    with open(out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"done: {seen} observations, {len(rows)} photos, {sum(r['screen_keep'] for r in rows)} with little green")


if __name__ == "__main__":
    main()
