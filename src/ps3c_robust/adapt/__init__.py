"""Label-shift adaptation utilities."""

from ps3c_robust.adapt.bbse import (
    confusion_matrix_joint,
    correct_probs,
    estimate_shift_weights,
    estimate_target_prior,
)

__all__ = [
    "confusion_matrix_joint",
    "estimate_shift_weights",
    "estimate_target_prior",
    "correct_probs",
]
