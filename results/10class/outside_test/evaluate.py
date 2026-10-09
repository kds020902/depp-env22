"""Score a model on 221 photos people posted online (Flickr, Wikimedia Commons), never used for training or tuning.

    python results/10class/outside_test/evaluate.py fetch --photos outside_photos
    python results/10class/outside_test/evaluate.py run --photos outside_photos [--model model.pth]

The labels in outside_test.csv were set by eye before any model was run on these photos.
"""

from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

import torch
from PIL import Image, ImageOps

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
from model_utils import input_size, load_model, make_transform, open_image  # noqa: E402

USER_AGENT = {"User-Agent": "depp-env22-student-project (university coursework)"}


def rows():
    with (HERE / "outside_test.csv").open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fetch(args):
    args.photos.mkdir(parents=True, exist_ok=True)
    for r in rows():
        out = args.photos / f"{r['id']}.jpg"
        if out.exists():
            continue
        try:
            with urllib.request.urlopen(urllib.request.Request(r["image_url"], headers=USER_AGENT), timeout=60) as resp:
                image = ImageOps.exif_transpose(Image.open(io.BytesIO(resp.read()))).convert("RGB")
        except OSError as error:
            print(f"could not fetch {r['id']}: {error}", file=sys.stderr)
            continue
        image.thumbnail((800, 800))  # the photos were scored at most 800 px on the long side
        image.save(out, quality=92)


def run(args):
    torch.set_num_threads(args.threads)
    model, checkpoint = load_model(args.model, torch.device("cpu"))
    transform, names = make_transform(input_size(checkpoint)), checkpoint["class_names"]
    items = [r for r in rows() if (args.photos / f"{r['id']}.jpg").exists()]
    preds = []
    with torch.inference_mode():
        for k in range(0, len(items), 32):
            batch = torch.stack([transform(open_image(args.photos / f"{r['id']}.jpg")) for r in items[k:k + 32]])
            preds += [names[i] for i in model(batch).argmax(1).tolist()]
    per = collections.defaultdict(lambda: [0, 0])
    for r, p in zip(items, preds):
        per[r["class_name"]][0] += p == r["class_name"]
        per[r["class_name"]][1] += 1
    result = {
        "model": str(args.model.resolve().relative_to(REPO)) if args.model.resolve().is_relative_to(REPO) else str(args.model),
        "photos": len(items),
        "correct": sum(p == r["class_name"] for r, p in zip(items, preds)),
        "species_correct": sum(p.split("_")[0] == r["class_name"].split("_")[0] for r, p in zip(items, preds)),
        "condition_correct": sum(p.split("_")[1] == r["class_name"].split("_")[1] for r, p in zip(items, preds)),
        "per_class": {c: f"{a}/{b}" for c, (a, b) in per.items()},
        "most_common_errors": [[t, p, n] for (t, p), n in collections.Counter(
            (r["class_name"], p) for r, p in zip(items, preds) if p != r["class_name"]).most_common(8)],
    }
    print(json.dumps(result, ensure_ascii=False, indent=1))
    if args.save:
        (HERE / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--photos", type=Path, default=Path("outside_photos"))
    r = sub.add_parser("run")
    r.add_argument("--photos", type=Path, default=Path("outside_photos"))
    r.add_argument("--model", type=Path, default=REPO / "model.pth")
    r.add_argument("--threads", type=int, default=4)
    r.add_argument("--save", action="store_true", help="write results.json next to this script")
    args = parser.parse_args()
    fetch(args) if args.command == "fetch" else run(args)


if __name__ == "__main__":
    main()
