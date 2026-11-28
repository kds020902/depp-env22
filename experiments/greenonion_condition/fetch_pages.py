"""Collect the photos shown on given web pages (e.g. pages agy found), for review by eye. No model picks photos.

    python experiments/greenonion_condition/fetch_pages.py PAGES.csv OUT_DIR

PAGES.csv columns: source_page, note. Redirects are followed. From each page: og:image and <img> sources (also data-src,
srcset), at most 20 per page, kept when the image opens and is at least 250 px on its short side. Writes OUT_DIR/manifest.csv.
These photos are usually copyrighted: they stay outside the repository and are used only to measure, never to train.
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from PIL import Image

UA = {"User-Agent": "veg-dataset-research/0.1 (student project; measurement only)"}


class Imgs(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.urls = []

    def handle_starttag(self, tag, attrs):
        a = {k: v for k, v in attrs if v}
        if tag == "meta" and (a.get("property") or a.get("name") or "").lower() in ("og:image", "twitter:image"):
            self.urls.append(a.get("content", ""))
        if tag in ("img", "source"):
            for k in ("data-src", "data-lazy-src", "data-original", "src"):
                if a.get(k):
                    self.urls.append(a[k]); break
            if a.get("srcset"):
                parts = [p.strip().split()[0] for p in a["srcset"].split(",") if p.strip()]
                if parts:
                    self.urls.append(parts[-1])


def clean(url):
    p = urlsplit(url)
    return urlunsplit((p.scheme, p.netloc, quote(p.path, safe="/%:@"), quote(p.query, safe="=&%:@/"), ""))


def get(url, limit):
    with urlopen(Request(clean(url), headers=UA), timeout=30) as r:
        return r.read(limit), r.geturl(), r.headers.get("Content-Type", "")


def main():
    pages, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for n, page in enumerate(csv.DictReader(open(pages, encoding="utf-8-sig", newline=""))):
        try:
            html, final, kind = get(page["source_page"], 3_000_000)
        except Exception as e:
            rows.append({"page_no": n, "page": page["source_page"][:120], "final_page": "", "note": page.get("note", ""), "file": "",
                         "image_url": "", "sha256": "", "status": f"page failed: {type(e).__name__}"}); continue
        parser = Imgs(); parser.feed(html.decode("utf-8", "replace"))
        urls = [urljoin(final, u) for u in dict.fromkeys(parser.urls) if u and not u.startswith("data:")]
        urls = [u for u in urls if not re.search(r"logo|icon|avatar|sprite|banner|\.svg|\.gif", u, re.I)][:20]
        kept = 0
        for u in urls:
            try:
                data, _, _ = get(u, 15_000_000)
                with Image.open(io.BytesIO(data)) as im:
                    w, h = im.size
                if min(w, h) < 250:
                    continue
                digest = hashlib.sha256(data).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                name = f"p{n:02d}_{kept:02d}.jpg"
                with Image.open(io.BytesIO(data)) as im:
                    im.convert("RGB").save(out / name, quality=92)
                rows.append({"page_no": n, "page": page["source_page"][:120], "final_page": final, "note": page.get("note", ""),
                             "file": name, "image_url": u, "sha256": digest, "status": "ok"})
                kept += 1
            except Exception:
                continue
            time.sleep(0.3)
        print(f"page {n}: {kept} photos from {urlsplit(final).netloc}", flush=True)
        time.sleep(1)
    with open(out / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
