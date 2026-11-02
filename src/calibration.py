"""Calibrated probabilities (temperature scaling) and the vegetable model's "not a trained vegetable" check."""

from __future__ import annotations

import torch
from torch import nn

from labels import CONDITION_NAMES, UNKNOWN_MESSAGE

def marginal_probabilities(probabilities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum the joint (species x condition) probabilities, rather than reusing joint argmax labels."""
    joint = probabilities.reshape(-1, probabilities.shape[1] // len(CONDITION_NAMES), len(CONDITION_NAMES))
    return joint.sum(dim=2), joint.sum(dim=1)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Temperature scaling: one number T so that softmax(logits / T) matches how often the model is right
    (fitted on the validation split by minimising cross-entropy)."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_t.detach().exp())


def unknown_score(logits: torch.Tensor) -> torch.Tensor:
    """Energy score, -logsumexp(logits): high when no trained class fits the photo well."""
    return -torch.logsumexp(logits, dim=1)


def fit_unknown_threshold(val_logits: torch.Tensor, accept_rate: float = 0.95,
                          extra_logits: torch.Tensor | None = None) -> float:
    """Threshold that still accepts accept_rate of the validation photos (all real vegetables),
    pooled with extra_logits: unlabeled real-world vegetable photos, when given."""
    logits = val_logits if extra_logits is None else torch.cat([val_logits, extra_logits])
    return float(torch.quantile(unknown_score(logits), accept_rate))


def calibrated_probabilities(logits: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    temperature = checkpoint.get("calibration", {}).get("temperature", 1.0)
    return (logits / temperature).softmax(1)


def is_unknown(logits: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    threshold = checkpoint.get("unknown_check", {}).get("threshold")
    if threshold is None:  # older checkpoints have no check
        return torch.zeros(len(logits), dtype=torch.bool)
    return unknown_score(logits) > threshold


def calibration_extras(val_logits: torch.Tensor, val_labels: torch.Tensor, accept_rate: float = 0.95,
                       extra_logits: torch.Tensor | None = None, extra_name: str = "") -> dict:
    """The checkpoint entries for calibrated probabilities and the unknown-photo check.
    The temperature is fitted on Valid only; the threshold on Valid, or on Valid pooled with extra real-world vegetable
    photos (extra_logits, no labels needed)."""
    threshold = fit_unknown_threshold(val_logits, accept_rate, extra_logits)
    check = {"method": "energy score -logsumexp(logits)", "threshold": threshold,
             "valid_accept_rate": accept_rate, "fitted_on": "val", "message": UNKNOWN_MESSAGE}
    if extra_logits is not None:
        check.update({"accept_rate": accept_rate, "fitted_on": f"val + {len(extra_logits)} {extra_name}".strip(),
                      "valid_accept_rate": float((unknown_score(val_logits) <= threshold).float().mean())})
    return {
        "calibration": {"method": "temperature scaling", "temperature": fit_temperature(val_logits, val_labels),
                        "fitted_on": "val"},
        "unknown_check": check,
    }
