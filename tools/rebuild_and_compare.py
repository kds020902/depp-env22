"""Rebuild the dataset with dataset/tools (v2, then v3) and compare the result with the committed dataset/.

Used to check that a change to dataset/tools keeps every photo assignment (docs/codex_handoff.md).

    # with the original Kaggle sources
    python tools/rebuild_and_compare.py --work WORK --src1 KAGGLE_MUHRIDDIN --src2 KAGGLE_SWOYAM
    # without them: stand-in sources from the repository photos + a mirror zip that holds the same files by name
    python tools/rebuild_and_compare.py --work WORK --mirror-zip muhammad0subhan_fruit-and-vegetable-disease-healthy-vs-rotten.zip

Stand-in sources (the Kaggle downloads returned 404 on 2026-11-19):
- 1st source: every candidate photo the build reads (1st-round manifest + review "passed" lists, 2,071 files). 1,800 come from
  dataset/ (byte-identical copies of the originals); 271 that were never kept come from the mirror zip by file name. For the
  1,800 in both, the mirror's bytes equal the repository's (checked here), which is the evidence for the other 271.
- 2nd source: the 60 sampled cucumbers, taken from dataset/ where they are already frame-cropped (not 450x320, so the cropper
  leaves them alone), plus three blank 450x320 frames so the cropper has a template.
Exit code 0 only when the rebuilt manifest.csv is byte-identical and every photo has the same relative path and SHA-256.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def stand_in_sources(work: Path, mirror_zip: Path) -> tuple[Path, Path]:
    from PIL import Image
    rows = list(csv.DictReader(open(REPO / "dataset/manifest.csv", encoding="utf-8-sig")))
    v2 = json.load(open(REPO / "dataset/review/v2_visual_review.json", encoding="utf-8"))["decisions"]
    v3 = json.load(open(REPO / "dataset/review/v3_visual_review.json", encoding="utf-8"))["decisions"]
    need = {r["source_archive_path"] for r in csv.DictReader(open(work / "v1_manifest.csv", encoding="utf-8-sig"))}
    need |= {p for k, v in v2.items() if not k.startswith("src2_") for p in v["passed"]}
    need |= {p for v in v3.values() for p in v["passed"]}
    repo = {r["source_archive_path"]: r for r in rows if r["source_dataset"] == "muhriddinmuxiddinov"}
    mirror = zipfile.ZipFile(mirror_zip)
    by_name = {n.split("/")[-1].lower(): n for n in mirror.namelist() if not n.endswith("/")}
    count = collections.Counter()
    for p in sorted(need):
        dst = work / "src1" / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        member = by_name.get(p.split("/")[-1].lower())
        if p in repo:
            shutil.copy2(REPO / "dataset" / repo[p]["image_path"], dst)
            if member is None or hashlib.sha256(mirror.read(member)).hexdigest() != repo[p]["sha256"]:
                count["repo photo differs from the mirror"] += 1
            count["from dataset/"] += 1
        else:
            dst.write_bytes(mirror.read(member))
            count["from the mirror by name"] += 1
    random.seed(2026)  # the sample build_dataset_v2.py draws
    sampled = [p for c in ("cucumber_fresh", "cucumber_rotten") for p in random.sample(v2["src2_" + c]["passed"], 30)]
    second = {r["source_archive_path"]: r for r in rows if r["source_dataset"] == "swoyam2609"}
    for p in sampled:
        dst = work / "src2" / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / "dataset" / second[p]["image_path"], dst)
        assert Image.open(dst).size != (450, 320)
    frames = work / "src2/dataset/frames_cucumber"
    frames.mkdir(parents=True, exist_ok=True)
    for k in range(3):
        Image.new("RGB", (450, 320), (200, 200, 200)).save(frames / f"frame{k}.png")
    print("stand-in sources:", dict(count))
    assert not count["repo photo differs from the mirror"]
    return work / "src1", work / "src2"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True, help="Empty working folder (outside the repository).")
    parser.add_argument("--src1", type=Path)
    parser.add_argument("--src2", type=Path)
    parser.add_argument("--mirror-zip", type=Path)
    parser.add_argument("--hash-seed", default="0", help="PYTHONHASHSEED for the builds (try a second value too).")
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    if any((work / d).exists() for d in ("v2", "v3")):  # photos left by an earlier build would be compared as if new
        raise SystemExit(f"{work} already holds a build (v2/ or v3/); use an empty folder for every run.")
    (work / "v1_manifest.csv").write_bytes(subprocess.run(["git", "show", "03db425:manifest.csv"], cwd=REPO, check=True,
                                                           capture_output=True).stdout)
    src1, src2 = (args.src1, args.src2) if args.src1 else stand_in_sources(work, args.mirror_zip)
    env = {**os.environ, "PYTHONHASHSEED": args.hash_seed}
    run = lambda *a: subprocess.run([sys.executable, *map(str, a)], cwd=REPO, check=True, env=env)
    run("dataset/tools/build_dataset_v2.py", "--src1", src1, "--src2", src2, "--v1-manifest", work / "v1_manifest.csv", "--out", work / "v2")
    run("dataset/tools/build_dataset_v3.py", "--src1", src1, "--base", work / "v2", "--out", work / "v3")
    print("--- 2nd round (first 1,200 rows) against dataset/")
    a = subprocess.run([sys.executable, "tools/compare_dataset_builds.py", work / "v2", "dataset", "--rows", "1200"], cwd=REPO).returncode
    print("--- 3rd round against dataset/")
    b = subprocess.run([sys.executable, "tools/compare_dataset_builds.py", work / "v3", "dataset"], cwd=REPO).returncode
    raise SystemExit(a or b)


if __name__ == "__main__":
    main()
