"""Compare two outputs of dataset/tools/build_dataset_v2.py / build_dataset_v3.py (before and after a refactor).

    python tools/compare_dataset_builds.py OUT_A OUT_B            # two build folders
    python tools/compare_dataset_builds.py OUT_A dataset --rows 1200   # against the committed dataset (first 1,200 rows = 2nd round)

Reports: whether manifest.csv is byte-identical; per column, how many rows differ; the photo assignment (class, split,
source path) as a set; and the relative path + SHA-256 list of the photo files. Exit code 0 only when everything is the same.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

PHOTO_DIRS = ("train", "valid", "test")


def manifest(folder: Path, rows: int | None):
    data = (folder / "manifest.csv").read_bytes()
    table = list(csv.DictReader(data.decode("utf-8-sig").splitlines()))
    return data, table[:rows] if rows else table


def photos(folder: Path, keep: set[str] | None):
    out = {}
    for d in PHOTO_DIRS:
        for p in sorted((folder / d).rglob("*")):
            rel = p.relative_to(folder).as_posix()
            if p.is_file() and (keep is None or rel in keep):
                out[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    parser.add_argument("--rows", type=int, help="Compare only the first N manifest rows (and their photos).")
    args = parser.parse_args()
    (raw_a, a), (raw_b, b) = manifest(args.a, args.rows), manifest(args.b, args.rows)
    same = True
    print("manifest.csv bytes identical:", raw_a == raw_b if not args.rows else "(not compared: --rows)")
    print("rows:", len(a), len(b))
    key = lambda r: (r["source_dataset"], r["source_archive_path"])
    assign = lambda t: {(key(r), r["class_name"], r["split"]) for r in t}
    only_a, only_b = assign(a) - assign(b), assign(b) - assign(a)
    print("photo assignment (source photo, class, split) identical:", not only_a and not only_b,
          f"| only in A {len(only_a)}, only in B {len(only_b)}")
    same &= not only_a and not only_b and len(a) == len(b)
    if len(a) == len(b):
        for col in a[0]:
            n = sum(x[col] != y.get(col) for x, y in zip(a, b))
            if n:
                same = False
                first = next((x["image_id"], x[col], y.get(col)) for x, y in zip(a, b) if x[col] != y.get(col))
                print(f"  column {col}: {n} rows differ, first {first}")
    keep = (lambda t: {r["image_path"].replace("\\", "/") for r in t})
    pa, pb = photos(args.a, keep(a) if args.rows else None), photos(args.b, keep(b) if args.rows else None)
    diff = sorted(k for k in pa.keys() | pb.keys() if pa.get(k) != pb.get(k))
    print("photo files (relative path + SHA-256) identical:", not diff, f"| files {len(pa)} / {len(pb)}, differing {len(diff)}")
    for k in diff[:5]:
        print("  ", k, (pa.get(k) or "missing")[:12], (pb.get(k) or "missing")[:12])
    same &= not diff
    if not args.rows:
        same &= raw_a == raw_b
    print("RESULT:", "SAME" if same else "DIFFERENT")
    raise SystemExit(0 if same else 1)


if __name__ == "__main__":
    main()
