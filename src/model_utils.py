"""Older entry point kept for the experiment scripts: what used to live here is now split into labels.py, images.py,
networks.py, calibration.py and decision.py. New code imports from those modules."""

from calibration import (
    calibrated_probabilities, calibration_extras, fit_temperature, fit_unknown_threshold, is_unknown, marginal_probabilities, unknown_score,
)
from decision import (
    Answers, clip_class_logits, clip_head_logits, clip_unknown_score, combined_logits, combined_probabilities, decide, is_unknown_combined, new_vegetable_decision, new_vegetable_logits, text_evidence, text_scores, zero_shot_logits,
)
from images import (
    IMAGE_EXTENSIONS, PREPROCESS, RandomLowRes, make_train_transform, make_transform, open_image, preprocess_for, read_image,
)
from labels import (
    CLASS_NAMES, CONDITION_NAMES, KOREAN, NO_CONDITION_SPECIES, SPECIES_NAMES, UNKNOWN_MESSAGE, UNTRAINED_CLASSES, condition_of,
)
from networks import (
    ARCHITECTURES, AvgMaxPool, HEATMAP_LAYERS, PRETRAINED_DIR, _apply_pooling, classifier_layer, create_model, forward_with_rot_maps, input_size, load_model, resolve_device, rot_map_overlay, set_classifier_layer, set_seed, state_checksum,
)

__all__ = [
    "calibrated_probabilities",
    "calibration_extras",
    "fit_temperature",
    "fit_unknown_threshold",
    "is_unknown",
    "marginal_probabilities",
    "unknown_score",
    "Answers",
    "clip_class_logits",
    "clip_head_logits",
    "clip_unknown_score",
    "combined_logits",
    "combined_probabilities",
    "decide",
    "is_unknown_combined",
    "new_vegetable_decision",
    "new_vegetable_logits",
    "text_evidence",
    "text_scores",
    "zero_shot_logits",
    "IMAGE_EXTENSIONS",
    "PREPROCESS",
    "RandomLowRes",
    "make_train_transform",
    "make_transform",
    "open_image",
    "preprocess_for",
    "read_image",
    "CLASS_NAMES",
    "CONDITION_NAMES",
    "KOREAN",
    "NO_CONDITION_SPECIES",
    "SPECIES_NAMES",
    "UNKNOWN_MESSAGE",
    "UNTRAINED_CLASSES",
    "condition_of",
    "ARCHITECTURES",
    "AvgMaxPool",
    "HEATMAP_LAYERS",
    "PRETRAINED_DIR",
    "_apply_pooling",
    "classifier_layer",
    "create_model",
    "forward_with_rot_maps",
    "input_size",
    "load_model",
    "resolve_device",
    "rot_map_overlay",
    "set_classifier_layer",
    "set_seed",
    "state_checksum",
]
