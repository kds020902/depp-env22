"""Add calibrated probabilities and the unknown-photo check to an existing checkpoint (fitted on Valid only).

train.py does this automatically; use this script for checkpoints trained before it did.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model_utils import CLASS_NAMES, calibration_extras, input_size, load_model
from train import CachedImages, predict_logits, read_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("model.pth"))
    parser.add_argument("--manifest", type=Path, default=Path("dataset/manifest.csv"))
    parser.add_argument("--accept-rate", type=float, default=0.95,
                        help="Share of validation photos the unknown-photo check must still accept.")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    model, checkpoint = load_model(args.model, torch.device("cpu"))
    manifest = args.manifest.expanduser().resolve()
    val = read_manifest(manifest, CLASS_NAMES[:len(checkpoint["class_names"])])["val"]
    loader = DataLoader(CachedImages(val, manifest.parent, train=False, size=input_size(checkpoint)), batch_size=32)
    logits, labels = predict_logits(model, loader, torch.device("cpu"))
    extras = calibration_extras(logits, labels, args.accept_rate)
    raw = torch.load(args.model, map_location="cpu", weights_only=True)
    raw.update(extras)
    torch.save(raw, args.model)
    print(extras)


if __name__ == "__main__":
    main()
