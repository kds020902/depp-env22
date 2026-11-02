"""The final answer: the vegetable model and the CLIP classifiers combined, turning away, and the new-vegetable switch.

decide() is the one place that turns model outputs into answers; predict.py, the tests and the experiment scripts use it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from calibration import calibrated_probabilities, is_unknown, unknown_score
from labels import CONDITION_NAMES


def text_scores(features: torch.Tensor, text: torch.Tensor, trained: torch.Tensor | None = None) -> torch.Tensor:
    """CLIP zero-shot class scores: 100 x cosine to each class's text prompts, centred over the classes per photo.
    With a `trained` mask the centre uses the trained classes only and untrained classes get 0."""
    z = 100 * features.float() @ text.to(features.device).float().T
    if trained is None:
        return z - z.mean(1, keepdim=True)
    return (z - z[:, trained].mean(1, keepdim=True)) * trained


# ---------- CLIP classifier head combined with the vegetable model (experiments/clip_classifier) ----------
def clip_head_logits(features: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    """Logits of the CLIP classifier head (multinomial logistic regression on L2-normalised CLIP image features).
    With "other_class" the last column is "not one of the five vegetables" (experiments/not_five_class)."""
    head = checkpoint["clip_head"]
    logits = features.float() @ head["coef"].T.to(features.device) + head["intercept"].to(features.device)
    if head.get("alpha"):
        # + alpha x CLIP zero-shot class scores on the class columns (experiments/improve_without_photos, as the V7 head):
        # 100 x cosine to each class's text prompts, centred over the classes per photo; "not five" column unchanged
        k = head["text_embeddings"].shape[0]
        logits = torch.cat([logits[:, :k] + head["alpha"] * text_scores(features, head["text_embeddings"]), logits[:, k:]], 1)
    return logits


def clip_class_logits(clip_logits: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    """The CLIP head's logits for the trained vegetable classes only."""
    return clip_logits[:, :-1] if checkpoint["clip_head"].get("other_class") else clip_logits


def text_evidence(features: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    """CLIP zero-shot evidence that the photo shows something other than the five vegetables (TV-12,
    experiments/zero_shot_reject): logsumexp(100 cos(photo, other words)) - logsumexp(100 cos(photo, five vegetables))."""
    words = checkpoint["clip_head"]["text_evidence"]
    f = features.float()
    other = 100 * f @ words["other"].to(f.device).float().T
    vegetables = 100 * f @ words["vegetables"].to(f.device).float().T
    return torch.logsumexp(other, 1) - torch.logsumexp(vegetables, 1)


def clip_unknown_score(clip_logits: torch.Tensor, checkpoint: dict, features: torch.Tensor | None = None) -> torch.Tensor:
    """Higher = less like a trained vegetable. With an "other" class: its log-odds against all vegetable classes;
    otherwise the energy of the head's logits, as for the vegetable model. When the head has "text_evidence" (TV-12),
    alpha x text_evidence() of the CLIP image features is added, so the features must be given."""
    head = checkpoint["clip_head"]
    if head.get("other_class"):
        score = clip_logits[:, -1] - torch.logsumexp(clip_logits[:, :-1], 1)
    else:
        score = unknown_score(clip_logits)
    if head.get("text_evidence"):
        if features is None:
            raise ValueError("this model's check also uses CLIP text evidence: pass the CLIP image features")
        score = score + head["text_evidence"]["alpha"] * text_evidence(features, checkpoint).to(score.dtype)
    return score


def combined_logits(logits: torch.Tensor, clip_logits: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    """Sum of the two models' log-probabilities (the vegetable model at its own temperature)."""
    temperature = checkpoint.get("calibration", {}).get("temperature", 1.0)
    return torch.log_softmax(logits / temperature, 1) + torch.log_softmax(clip_class_logits(clip_logits, checkpoint), 1)


def combined_probabilities(combined: torch.Tensor, checkpoint: dict) -> torch.Tensor:
    return (combined / checkpoint["clip_head"]["temperature"]).softmax(1)


def is_unknown_combined(logits: torch.Tensor, clip_logits: torch.Tensor, checkpoint: dict,
                        features: torch.Tensor | None = None) -> torch.Tensor:
    """Turned away when the CLIP head's check says the photo is not one of the trained vegetables, and also when the
    vegetable model's own check does, unless the head turns that off ("vegetable_model_check": False, experiments/clip_only_reject).
    `features` (CLIP image features) are needed when the head uses text evidence (TV-12)."""
    clip_says = clip_unknown_score(clip_logits, checkpoint, features) > checkpoint["clip_head"]["unknown_threshold"]
    if not checkpoint["clip_head"].get("vegetable_model_check", True):
        return clip_says
    return is_unknown(logits, checkpoint) | clip_says


# ---------- onion, sweet potato, green onion: a second CLIP head (experiments/new_vegetables, V7 = TV-10) ----------
def zero_shot_logits(features: torch.Tensor, head: dict) -> torch.Tensor:
    """CLIP zero-shot class scores (100 x cosine to each class's text prompts), centred over the trained classes per photo;
    classes without training photos get 0."""
    trained = torch.tensor([c not in head.get("untrained_classes", []) for c in head["class_names"]], device=features.device)
    return text_scores(features, head["text_embeddings"], trained)


def new_vegetable_logits(features: torch.Tensor, head: dict) -> torch.Tensor:
    """The second head's logits (16 classes + "not one of the eight vegetables"): logistic regression on CLIP image features,
    plus alpha x zero-shot class scores on the 16 class columns when the head has text embeddings (V7 = TV-10)."""
    z = features.float() @ head["coef"].T.to(features.device) + head["intercept"].to(features.device)
    if head.get("alpha"):
        k = len(head["class_names"])
        z = torch.cat([z[:, :k] + head["alpha"] * zero_shot_logits(features, head), z[:, k:]], 1)
    return z


def new_vegetable_decision(probabilities: torch.Tensor, unknown: torch.Tensor, features: torch.Tensor,
                           checkpoint: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The five-vegetable system answers first; the second CLIP head (16 classes + "not one of the eight vegetables")
    may change the answer to onion, sweet potato or green onion:
    - a photo the five-vegetable system turned away, when the head does not turn it away and its most likely kind
      (sum of the fresh and rotten probabilities) is a new vegetable;
    - a photo the five-vegetable system answered, only when the head gives the new vegetables together at least
      `switch_probability` and its most likely kind is a new vegetable.
    Returns probabilities over the head's 16 classes (the head's, at its display temperature, for switched photos;
    otherwise the five-vegetable probabilities padded with zeros), the chosen class per photo (for a switched photo the
    head's most likely kind, then its more likely condition), the new unknown flags and which photos were switched."""
    head = checkpoint["new_vegetable_head"]
    z = new_vegetable_logits(features, head)
    k = z.shape[1] - 1                                    # last column: not one of the eight vegetables
    p = z[:, :k].softmax(1)
    n, n_species = len(z), k // len(CONDITION_NAMES)
    species = p.reshape(n, n_species, len(CONDITION_NAMES)).sum(2).argmax(1)
    within = z[:, :k].reshape(n, n_species, len(CONDITION_NAMES))[torch.arange(n), species].argmax(1)
    head_index = species * len(CONDITION_NAMES) + within
    head_unknown = z[:, k] - torch.logsumexp(z[:, :k], 1) > head["unknown_threshold"]
    first_new = head["first_new_species"]
    is_new = species >= first_new
    new_share = p[:, first_new * len(CONDITION_NAMES):].sum(1)
    switch = (unknown & ~head_unknown & is_new) | (~unknown & (new_share >= head["switch_probability"]) & is_new)
    padded = torch.zeros(n, k, dtype=probabilities.dtype)
    padded[:, :probabilities.shape[1]] = probabilities
    head_probabilities = (z[:, :k] / head["temperature"]).softmax(1).to(padded.dtype)
    output = torch.where(switch[:, None], head_probabilities, padded)
    index = torch.where(switch, head_index, padded.argmax(1))
    return output, index, unknown & ~switch, switch


@dataclass
class Answers:
    """Answers for a batch of photos (one row each)."""
    probabilities: torch.Tensor   # shown probabilities over the classes (16 with the new-vegetable head, else 10)
    index: torch.Tensor           # chosen class
    probability: torch.Tensor     # probability of the chosen class
    unknown: torch.Tensor         # turned away ("학습된 데이터가 아닙니다")
    switched: torch.Tensor        # answered by the new-vegetable head (onion, sweet potato, green onion)
    five: torch.Tensor            # five-vegetable probabilities (10 classes)
    five_unknown: torch.Tensor    # turned away by the five-vegetable system, before the new-vegetable head


def decide(logits: torch.Tensor, checkpoint: dict, features: torch.Tensor | None = None, new_vegetables: bool = True) -> Answers:
    """The answer of the whole system for a batch: `logits` from the vegetable model, `features` the CLIP image features
    (None = vegetable model alone, as predict.py --no-clip). The new-vegetable head is used when the checkpoint has it,
    features are given and new_vegetables is True."""
    if features is not None and "clip_head" in checkpoint:
        clip_logits = clip_head_logits(features, checkpoint)
        five = combined_probabilities(combined_logits(logits, clip_logits, checkpoint), checkpoint)
        five_unknown = is_unknown_combined(logits, clip_logits, checkpoint, features)
    else:
        five = calibrated_probabilities(logits, checkpoint)
        five_unknown = is_unknown(logits, checkpoint)
    if new_vegetables and features is not None and "new_vegetable_head" in checkpoint and "clip_head" in checkpoint:
        probabilities, index, unknown, switched = new_vegetable_decision(five, five_unknown, features, checkpoint)
    else:
        probabilities, index = five, five.argmax(1)
        unknown, switched = five_unknown, torch.zeros_like(five_unknown)
    probability = probabilities[torch.arange(len(index)), index]
    return Answers(probabilities, index, probability, unknown, switched, five, five_unknown)


# ---------- holding an answer back (experiments/improve_without_photos, part 4) ----------
# The pre-registered rule picked 0.9: on the outside development photos the answers given at 0.9 or more were 96.3% right
# (91.6% when every answer is given), and confidently wrong answers fell from 34 to 12.
HOLD_BACK_BELOW = 0.9


def held_back(a: Answers, below: float = HOLD_BACK_BELOW) -> torch.Tensor:
    """Answers given with too little confidence to show as an answer: not turned away and probability below `below`."""
    return ~a.unknown & (a.probability < below)


def fresh_or_rotten_doubt(a: Answers) -> torch.Tensor:
    """The two most likely classes are the fresh and the rotten class of one vegetable."""
    top2 = a.probabilities.topk(2, dim=1).indices
    return top2[:, 0] // len(CONDITION_NAMES) == top2[:, 1] // len(CONDITION_NAMES)
