"""Fit the CLIP classifier head and store it in a checkpoint (experiments/clip_classifier).

    python src/train_clip_head.py --model model.pth --extra-photos lvis_vegetables

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
from pathlib import Path

import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from clip_model import ARCH, PRETRAINED, ClipModel
from model_utils import (
    CLASS_NAMES, IMAGE_EXTENSIONS, combined_logits, fit_temperature, input_size, load_model, open_image, unknown_score,
)
from train import CachedImages, predict_logits, read_manifest


def clip_features(clip: ClipModel, paths: list[Path], batch: int = 32) -> torch.Tensor:
    return torch.cat([clip.image_features([open_image(p) for p in paths[i:i + batch]]).cpu()
                      for i in range(0, len(paths), batch)])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=Path("model.pth"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset/manifest.csv"))
    parser.add_argument("--extra-photos", type=Path, required=True,
                        help="Folder of real-world vegetable photos pooled with Valid for the CLIP head's threshold "
                             "(experiments/real_scene_threshold/crop_lvis.py makes the 192 used for model.pth).")
    parser.add_argument("--c", type=float, default=10.0, help="Inverse regularisation strength of the head.")
    parser.add_argument("--accept-rate", type=float, default=0.95)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, checkpoint = load_model(args.model, torch.device("cpu"))
    manifest = args.manifest.expanduser().resolve()
    splits = read_manifest(manifest, CLASS_NAMES[:len(checkpoint["class_names"])])
    clip = ClipModel("cpu")

    train_x = clip_features(clip, [manifest.parent / r["image_path"] for r in splits["train"]])
    train_y = [int(r["class_id"]) for r in splits["train"]]
    head = LogisticRegression(C=args.c, max_iter=5000).fit(train_x.numpy(), train_y)
    clip_head = {"clip": {"arch": ARCH, "pretrained": PRETRAINED},
                 "coef": torch.tensor(head.coef_, dtype=torch.float32),
                 "intercept": torch.tensor(head.intercept_, dtype=torch.float32)}
    with_head = {**checkpoint, "clip_head": clip_head}

    val = splits["val"]
    loader = DataLoader(CachedImages(val, manifest.parent, train=False, size=input_size(checkpoint)), batch_size=32)
    with torch.no_grad():
        val_logits, val_y = predict_logits(model, loader, torch.device("cpu"))
    val_clip = clip_features(clip, [manifest.parent / r["image_path"] for r in val]) @ clip_head["coef"].T + clip_head["intercept"]
    temperature = fit_temperature(combined_logits(val_logits, val_clip, checkpoint), val_y)

    extra = sorted(p for p in args.extra_photos.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    extra_clip = clip_features(clip, extra) @ clip_head["coef"].T + clip_head["intercept"]
    threshold = float(torch.quantile(unknown_score(torch.cat([val_clip, extra_clip])), args.accept_rate))

    clip_head.update({"head": f"multinomial logistic regression, C={args.c}, on {len(train_y)} Train photos",
                      "temperature": temperature, "temperature_fitted_on": "val (combined model)",
                      "unknown_threshold": threshold, "accept_rate": args.accept_rate,
                      "unknown_fitted_on": f"val + {len(extra)} extra real-world vegetable photos",
                      "notes": "experiments/clip_classifier"})
    raw = torch.load(args.model, map_location="cpu", weights_only=True)
    raw["clip_head"] = clip_head
    torch.save(raw, args.model)
    combined = combined_logits(val_logits, val_clip, with_head)
    print({"train_photos": len(train_y), "valid_accuracy_combined": int((combined.argmax(1) == val_y).sum()),
           "valid_accuracy_clip_head": int((val_clip.argmax(1) == val_y).sum()), "temperature": round(temperature, 4),
           "clip_unknown_threshold": round(threshold, 4)})


if __name__ == "__main__":
    main()
