"""Fit the CLIP classifier head and store it in a checkpoint (experiments/clip_classifier).

    python src/train_clip_head.py --model model.pth --extra-photos lvis_vegetables
    python src/train_clip_head.py --model model.pth --extra-photos lvis_vegetables --extra-train extra_train.csv --clip-check-only

--extra-train adds photos to the head's training set (CSV with columns path, class_name). class_name "not_five" photos
(fruit, everyday photos, other vegetables) become an 11th class, and the head's "not a trained vegetable" score becomes
that class's log-odds against all vegetable classes (experiments/not_five_class).

1. CLIP image features (open_clip ViT-B-16, DataComp-XL weights) for the Train and Valid photos of the manifest.
2. Multinomial logistic regression (C=10) on the Train photos only.
3. Temperature of the combined model (vegetable model log-probabilities + CLIP head log-probabilities), fitted on Valid.
4. The CLIP head's own "not a trained vegetable" threshold: 95% of Valid plus the extra real-world vegetable photos pass,
   the same rule as the vegetable model's threshold (src/calibrate.py --extra-photos).
predict.py then combines both models and turns a photo away when either model's check says so.
Needs scikit-learn and open_clip_torch.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from clip_model import ARCH, PRETRAINED, ClipModel
from model_utils import (
    CLASS_NAMES, IMAGE_EXTENSIONS, clip_unknown_score, combined_logits, fit_temperature, input_size, load_model, open_image,
)

NOT_FIVE = "not_five"
from train import CachedImages, predict_logits, read_manifest
from watermark_aug import Dice, stamp


def clip_features(clip: ClipModel, paths: list[Path], batch: int = 32, cache: Path | None = None,
                  keys: list[str] | None = None, load=None) -> torch.Tensor:
    """CLIP image features, one row per photo. With `cache` (an .npz file) finished batches are saved as they go, so a run
    that is stopped can continue where it stopped. `keys`/`load` name and open photos that are not plain files
    (the watermarked copies)."""
    keys = keys or [str(p) for p in paths]
    load = load or (lambda i: open_image(paths[i]))
    store = {}
    if cache is not None and cache.exists():
        with np.load(cache) as saved:
            store = dict(zip(saved["keys"].tolist(), saved["features"]))
    todo = [i for i, k in enumerate(keys) if k not in store]
    for n, start in enumerate(range(0, len(todo), batch)):
        chunk = todo[start:start + batch]
        store.update(zip([keys[i] for i in chunk], clip.image_features([load(i) for i in chunk]).cpu().numpy()))
        if cache is not None and (n % 16 == 15 or start + batch >= len(todo)):
            np.savez(cache, keys=np.array(list(store)), features=np.stack(list(store.values())))
    return torch.tensor(np.stack([store[k] for k in keys]))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("model.pth"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset/manifest.csv"))
    parser.add_argument("--extra-photos", type=Path, required=True,
                        help="Folder of real-world vegetable photos pooled with Valid for the CLIP head's threshold "
                             "(experiments/real_scene_threshold/crop_lvis.py makes the 192 used for model.pth).")
    parser.add_argument("--extra-train", type=Path, help="CSV (path, class_name) of extra training photos for the head; "
                        "class_name may be a vegetable class or not_five.")
    parser.add_argument("--clip-check-only", action="store_true",
                        help="Turn photos away by the head's own check only, not also by the vegetable model's energy check "
                             "(experiments/clip_only_reject; meant for a head with the not_five class).")
    parser.add_argument("--watermark-fresh", type=float, default=0.0,
                        help="Add one copy of every training photo; the copy of a fresh photo gets a synthetic stock-photo "
                             "watermark with this chance (experiments/watermark). 0 for all three = no copies.")
    parser.add_argument("--watermark-rotten", type=float, default=0.0, help="The same for rotten photos.")
    parser.add_argument("--watermark-other", type=float, default=0.0, help="The same for not_five photos.")
    parser.add_argument("--feature-cache", type=Path,
                        help="Optional .npz file that keeps the CLIP features computed so far (resume a stopped run).")
    parser.add_argument("--c", type=float, default=10.0, help="Inverse regularisation strength of the head.")
    parser.add_argument("--accept-rate", type=float, default=0.95)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, checkpoint = load_model(args.model, torch.device("cpu"))
    manifest = args.manifest.expanduser().resolve()
    splits = read_manifest(manifest, CLASS_NAMES[:len(checkpoint["class_names"])])
    clip = ClipModel("cpu")

    class_names = list(checkpoint["class_names"])
    train_paths = [manifest.parent / r["image_path"] for r in splits["train"]]
    train_y = [int(r["class_id"]) for r in splits["train"]]
    n_extra = 0
    if args.extra_train:
        with args.extra_train.open(encoding="utf-8") as handle:
            for r in csv.DictReader(handle):
                train_paths.append(Path(r["path"]))
                train_y.append(len(class_names) if r["class_name"] == NOT_FIVE else class_names.index(r["class_name"]))
                n_extra += 1
    other_class = len(class_names) in train_y
    cache = args.feature_cache
    train_x = clip_features(clip, train_paths, cache=cache)
    n_stamped = 0
    if max(args.watermark_fresh, args.watermark_rotten, args.watermark_other) > 0:
        # One copy of every photo, so the classes keep their sizes. Stamping fresh copies more often than rotten ones makes a
        # watermark about as common in both (real watermarks: ~8% of fresh, ~47% of rotten photos in Valid + Test).
        generator = torch.Generator().manual_seed(0)
        chance = lambda y: (args.watermark_other if y == len(class_names) else
                            args.watermark_rotten if y % 2 else args.watermark_fresh)
        stamped = [i for i, y in enumerate(train_y) if float(torch.rand(1, generator=generator)) < chance(y)]
        # each copy's watermark comes from its own seed, so the result does not depend on the order or on a resumed run
        copies = train_x.clone()
        copies[stamped] = clip_features(
            clip, [], cache=cache, keys=[f"watermark:{i}:{train_paths[i]}" for i in stamped],
            load=lambda k: stamp(open_image(train_paths[stamped[k]]), Dice(torch.Generator().manual_seed(stamped[k] + 1))))
        train_x, train_y, n_stamped = torch.cat([train_x, copies]), train_y + train_y, len(stamped)
    head = LogisticRegression(C=args.c, max_iter=5000).fit(train_x.numpy(), train_y)
    # A class with no training photos (UNTRAINED_CLASSES) gets a zero row and a very low intercept, so it is never chosen.
    n_out = len(class_names) + int(other_class)
    coef = torch.zeros(n_out, train_x.shape[1]); intercept = torch.full((n_out,), float(head.intercept_.min()) - 50.0)
    coef[torch.tensor(head.classes_)] = torch.tensor(head.coef_, dtype=torch.float32)
    intercept[torch.tensor(head.classes_)] = torch.tensor(head.intercept_, dtype=torch.float32)
    clip_head = {"clip": {"arch": ARCH, "pretrained": PRETRAINED}, "coef": coef, "intercept": intercept, "other_class": other_class}
    with_head = {**checkpoint, "clip_head": clip_head}

    val = splits["val"]
    loader = DataLoader(CachedImages(val, manifest.parent, train=False, size=input_size(checkpoint)), batch_size=32)
    with torch.no_grad():
        val_logits, val_y = predict_logits(model, loader, torch.device("cpu"))
    val_clip = clip_features(clip, [manifest.parent / r["image_path"] for r in val], cache=cache) @ clip_head["coef"].T + clip_head["intercept"]
    temperature = fit_temperature(combined_logits(val_logits, val_clip, with_head), val_y)

    extra = sorted(p for p in args.extra_photos.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    extra_clip = clip_features(clip, extra, cache=cache) @ clip_head["coef"].T + clip_head["intercept"]
    threshold = float(torch.quantile(clip_unknown_score(torch.cat([val_clip, extra_clip]), with_head), args.accept_rate))

    n_photos = len(train_y) // 2 if n_stamped else len(train_y)
    n_other = sum(y == len(class_names) for y in train_y[:n_photos])
    clip_head.update({"head": f"multinomial logistic regression, C={args.c}, on {n_photos - n_extra} Train photos"
                              + (f" + {n_extra} extra photos ({n_other} not_five)" if n_extra else "")
                              + (f"; plus one copy of each, {n_stamped} of them with a synthetic watermark "
                                 f"(chance fresh {args.watermark_fresh}, rotten {args.watermark_rotten}, "
                                 f"not_five {args.watermark_other})" if n_stamped else ""),
                      "temperature": temperature, "temperature_fitted_on": "val (combined model)",
                      "unknown_threshold": threshold, "accept_rate": args.accept_rate,
                      "unknown_fitted_on": f"val + {len(extra)} extra real-world vegetable photos",
                      "vegetable_model_check": not args.clip_check_only,
                      "notes": "experiments/not_five_class" if other_class else "experiments/clip_classifier"})
    raw = torch.load(args.model, map_location="cpu", weights_only=True)
    raw["clip_head"] = clip_head
    torch.save(raw, args.model)
    combined = combined_logits(val_logits, val_clip, with_head)
    print({"train_photos": n_photos, "watermarked_copies": n_stamped, "valid_accuracy_combined": int((combined.argmax(1) == val_y).sum()),
           "valid_accuracy_clip_head": int((val_clip[:, :len(class_names)].argmax(1) == val_y).sum()), "temperature": round(temperature, 4),
           "clip_unknown_threshold": round(threshold, 4)})


if __name__ == "__main__":
    main()
