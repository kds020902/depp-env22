"""Build the onion / sweet potato / green onion head ("new_vegetable_head") and put it into a checkpoint.

    python src/train_new_veg_head.py --model model.pth --data experiments/new_vegetables/v7/head_data.npz

The head is a multinomial logistic regression (C=10) on CLIP image features (ViT-B-16, DataComp-XL, L2-normalised) with 16
vegetable classes and a 17th "not one of the eight vegetables" class. With --alpha > 0, alpha x CLIP zero-shot class scores
(text prompts stored in the data file) are added to the 16 class logits (experiments/new_vegetables, V7 = TV-10).

head_data.npz holds every training feature and label (so the photos are not needed) and the development sets used to set
- the "not eight" threshold: 95% of Valid (old + new vegetables) and of 192 real-scene vegetable crops pass;
- the switch probability tau: the smallest of 0.5 / 0.7 / 0.8 / 0.9 / 0.95 that moves at most one old-vegetable development photo
  the five-vegetable system answered (Valid, internet 221, the 160 exam, Asian cucumbers, real-scene crops) to a new vegetable;
- a display temperature (probabilities shown for switched photos only), fitted on Valid.
The five-vegetable part of the checkpoint (vegetable model + CLIP head) is not changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model_utils import (
    CLASS_NAMES, UNTRAINED_CLASSES, clip_head_logits, combined_logits, combined_probabilities, fit_temperature,
    is_unknown_combined, new_vegetable_decision, new_vegetable_logits,
)

TAU_GRID = (0.5, 0.7, 0.8, 0.9, 0.95)
OLD_DEV = ("valid", "odev", "o160", "asian37", "lvis")


def fit_head(x: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=10, max_iter=5000).fit(x, y)
    coef = np.zeros((k + 1, x.shape[1])); intercept = np.full(k + 1, lr.intercept_.min() - 50.0)  # untrained classes stay off
    coef[lr.classes_] = lr.coef_; intercept[lr.classes_] = lr.intercept_
    return coef, intercept


def five_vegetable_answer(checkpoint: dict, b: np.ndarray, f: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    logits, feats = torch.tensor(b), torch.tensor(f)
    clip_logits = clip_head_logits(feats, checkpoint)
    probabilities = combined_probabilities(combined_logits(logits, clip_logits, checkpoint), checkpoint)
    return probabilities, is_unknown_combined(logits, clip_logits, checkpoint, feats)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True, help="Checkpoint with the five-vegetable system (model.pth).")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.25, help="Weight of the CLIP zero-shot class scores (0 = V6-H, TV-9).")
    parser.add_argument("--out", type=Path, help="Where to write the checkpoint (default: overwrite --model).")
    args = parser.parse_args()
    data = np.load(args.data)
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    k = len(CLASS_NAMES)
    coef, intercept = fit_head(data["train_X"], data["train_y"].astype(int), k)
    head = {
        "clip": dict(checkpoint["clip_head"]["clip"]), "class_names": list(CLASS_NAMES), "first_new_species": 5,
        "coef": torch.tensor(coef, dtype=torch.float32), "intercept": torch.tensor(intercept, dtype=torch.float32),
        "other_name": "not one of the eight vegetables", "untrained_classes": sorted(UNTRAINED_CLASSES), "no_condition_species": ["greenonion"],
        "alpha": float(args.alpha), "temperature": 1.0, "switch_probability": 1.0, "unknown_threshold": 0.0,
    }
    if args.alpha:
        head["text_embeddings"] = torch.tensor(data["text16"], dtype=torch.float32)
        head["prompts"] = [str(p) for p in data["prompts"]]
    feats = lambda name: torch.tensor(data[f"dev_{name}_X"])

    # "not eight" threshold: 95% of Valid (old + new) and the real-scene crops pass
    def not_eight(f):
        z = new_vegetable_logits(f, head)
        return z[:, k] - torch.logsumexp(z[:, :k], 1)
    head["unknown_threshold"] = float(torch.quantile(torch.cat([not_eight(feats(n)) for n in ("valid", "newval", "lvis")]), 0.95))

    # tau: old-vegetable development photos the five-vegetable system answered, moved to a new vegetable
    def moved(tau: float) -> int:
        head["switch_probability"] = tau
        trial = {**checkpoint, "new_vegetable_head": head}
        count = 0
        for name in OLD_DEV:
            probabilities, unknown = five_vegetable_answer(checkpoint, data[f"dev_{name}_b"], data[f"dev_{name}_X"])
            _, index, still_unknown, switched = new_vegetable_decision(probabilities, unknown, feats(name), trial)
            count += int((~unknown & switched).sum())
        return count
    grid = {tau: moved(tau) for tau in TAU_GRID}
    head["switch_probability"] = min([tau for tau, m in grid.items() if m <= 1] or [TAU_GRID[-1]])

    # display temperature (switched photos only), fitted on Valid old + new
    z = torch.cat([new_vegetable_logits(feats(n), head)[:, :k] for n in ("valid", "newval")])
    labels = torch.tensor(np.concatenate([data["dev_valid_y"], data["dev_newval_y"]]).astype(int))
    head["temperature"] = fit_temperature(z.detach(), labels)

    counts = {c: int((data["train_y"] == i).sum()) for i, c in enumerate(CLASS_NAMES) if i >= 10}
    head.update({
        "head": f"multinomial logistic regression, C=10, 17 outputs, {len(data['train_y'])} training rows; new-vegetable Train {counts}"
                + (f"; + {args.alpha} x CLIP zero-shot class scores" if args.alpha else ""),
        "unknown_fitted_on": "95% of Valid (old + new vegetables) and 192 real-scene vegetable crops pass",
        "switch_rule": f"smallest tau moving at most 1 old-vegetable development photo; moved per tau: {grid}",
        "temperature_fitted_on": "Valid (old + new vegetables), display only",
        "data": str(args.data),
    })
    checkpoint["new_vegetable_head"] = head
    out = args.out or args.model
    torch.save(checkpoint, out)
    print(f"saved {out}: not-eight threshold {head['unknown_threshold']:.4f}, tau {head['switch_probability']} {grid}, "
          f"display temperature {head['temperature']:.4f}")


if __name__ == "__main__":
    main()
