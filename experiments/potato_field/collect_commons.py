"""Download potato candidate photos from Wikimedia Commons (no labels yet; labelling is by eye afterwards).

    python experiments/potato_field/collect_commons.py OUT_DIR

Sources: files in Category:Potato_harvest and Category:Potato_blight_caused_by_Phytophthora_infestans, plus files agy listed.
Only freely licensed files (license name from Commons metadata is recorded). Writes OUT_DIR/manifest.csv with
file, title, source_url, image_url, author (uploader), license, sha256, width, height, source_group (= "commons:" + uploader).
Images stay outside the repository.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

API = "https://commons.wikimedia.org/w/api.php"
UA = {"User-Agent": "veg-dataset-research/0.1 (student project)"}
CATS = ["Category:Potato_harvest", "Category:Potato_blight_caused_by_Phytophthora_infestans"]
EXTRA = ["File:Potato_soft_rot_(5684245277).jpg", "File:Rotten_potatoes.JPG", "File:Tuber_infected_by_P._infestans.jpg",
         "File:Potato_Late_Blight.JPG", "File:Late_blight_on_potato.jpg", "File:Phytophtora_infestans-effects.jpg"]
FREE = re.compile(r"^(CC0|CC BY|CC BY-SA|Public domain|PD)", re.I)


def api(**params):
    params.update(format="json")
    with urlopen(Request(API + "?" + urlencode(params), headers=UA), timeout=30) as r:
        return json.load(r)


def members(cat):
    out, cont = [], {}
    while True:
        d = api(action="query", list="categorymembers", cmtitle=cat, cmtype="file", cmlimit=500, **cont)
        out += [m["title"] for m in d["query"]["categorymembers"]]
        if "continue" not in d:
            return out
        cont = {"cmcontinue": d["continue"]["cmcontinue"]}


def info(titles):
    d = api(action="query", prop="imageinfo", iiprop="url|size|user|extmetadata|mime", iiurlwidth=1280, titles="|".join(titles))
    return [p for p in d["query"]["pages"].values() if "imageinfo" in p]


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    titles = list(dict.fromkeys(sum((members(c) for c in CATS), []) + [t.replace("_", " ") for t in EXTRA]))
    old = list(csv.DictReader(open(out / "manifest.csv", encoding="utf-8"))) if (out / "manifest.csv").exists() else []
    done = {r["title"] for r in old}
    rows = old                       # resume: files already saved are kept, their titles skipped
    titles = [t for t in titles if t not in done]
    for i in range(0, len(titles), 40):
        for p in info(titles[i:i + 40]):
            ii = p["imageinfo"][0]
            lic = ii.get("extmetadata", {}).get("LicenseShortName", {}).get("value", "")
            if not ii.get("mime", "").startswith("image/jpeg") or not FREE.match(lic):
                continue
            url = ii.get("thumburl") or ii["url"]
            name = f"commons_{len(rows):04d}.jpg"
            try:
                with urlopen(Request(url, headers=UA), timeout=60) as r:
                    data = r.read()
            except Exception as e:  # keep going; record nothing for a failed file
                print("skip", p["title"], e)
                continue
            (out / name).write_bytes(data)
            rows.append({"file": name, "title": p["title"], "source_url": "https://commons.wikimedia.org/wiki/" + quote(p["title"].replace(" ", "_")),
                         "image_url": url, "author": ii.get("user", ""), "license": lic, "sha256": hashlib.sha256(data).hexdigest(),
                         "width": ii.get("thumbwidth") or ii.get("width"), "height": ii.get("thumbheight") or ii.get("height"),
                         "source_group": "commons:" + ii.get("user", "")})
            time.sleep(1.5)  # be gentle: Commons answers 429 to fast clients
    with open(out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(titles)} titles, {len(rows)} free JPEGs saved to {out}")


if __name__ == "__main__":
    main()
