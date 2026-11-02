"""Download the internet photos of a photo list (the blind / outside test sets keep photos out of git for licence reasons).

    python tools/fetch_listed_photos.py results/t3_test/t3_test.csv  $DEPP_WORK/t3/img
    python tools/fetch_listed_photos.py results/t4_test/t4_test.csv  $DEPP_WORK/t4/img
    python tools/fetch_listed_photos.py results/10class/outside_test2/outside_test2.csv  $DEPP_WORK/weak/t2img

Saves <id>.jpg per row. Uses the row's image_url when there is one; iNaturalist photos ("inat-<photo id>") from the public
open-data bucket; Wikimedia Commons photos from their file page. Photos that were deleted upstream are reported, not guessed.
Check the SHA-256 against the working-folder copy if you have one; a re-encoded upstream file can differ in bytes.
"""

import csv
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = {"User-Agent": "depp-env22 photo list fetcher (university coursework)"}


def candidates(row: dict) -> list[str]:
    if row.get("image_url"):
        return [row["image_url"]]
    if row["id"].startswith("inat-"):
        pid = row["id"][5:]
        return [f"https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/medium.{ext}" for ext in ("jpg", "jpeg", "png")]
    if "commons.wikimedia.org" in row.get("page", ""):
        title = urllib.parse.unquote(row["page"].rsplit("/", 1)[-1]).replace("File:", "")
        return [f"https://commons.wikimedia.org/wiki/Special:FilePath/{urllib.parse.quote(title)}?width=800"]
    return []


def fetch(url: str, dst: Path, tries: int = 4) -> bool:
    """True when saved. Waits and retries on rate limits (HTTP 429/503); gives up on 404 (deleted upstream)."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as resp:
                dst.write_bytes(resp.read())
            return True
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503):
                return False
            wait = e.headers.get("retry-after")
            time.sleep(int(wait) + 1 if wait and wait.isdigit() else 10 * (k + 1))
        except Exception:
            time.sleep(3 * (k + 1))
    return False


def main(list_csv: str, out_dir: str, limit: int | None = None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(list_csv, encoding="utf-8")))[:limit]
    missing = []
    for r in rows:
        dst = out / f"{r['id']}.jpg"
        if dst.exists() and dst.stat().st_size > 2000:
            continue
        if not any(fetch(url, dst) for url in candidates(r)):
            missing.append(r["id"])
    print(f"{len(rows) - len(missing)}/{len(rows)} photos in {out}; missing: {missing[:20]}{' ...' if len(missing) > 20 else ''}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
