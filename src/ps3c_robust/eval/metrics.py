"""Metrics used across all three stages.

Two numbers carry the paper:

* `distribution_shift_drop` — the test→eval F1 collapse that the original
  PS3C challenge reported and that this work sets out to mitigate.
* `expected_calibration_error` — the calibration metric the original
  challenge does **not** report. The BSc thesis showed ECE rising 64×
  under medical domain shift (0.014 → 0.889 on PathMNIST → DermaMNIST);
  recovering ECE under the PS3C preprocessing shift is one of the
  contributions of this paper.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 — the headline number from the original paper."""
    return float(f1_score(y_true, y_pred, average="macro"))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Top-1 accuracy as a fraction in [0, 1]."""
    if len(y_true) == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Expected Calibration Error with equal-width binning (Guo et al., 2017).

    Lower is better. Measures the gap between predicted confidence and
    observed accuracy across `n_bins` confidence bins:

        ECE = Σ_m (|B_m| / N) · |acc(B_m) - conf(B_m)|

    This is a NumPy port of the developer's BSc thesis implementation
    (`src/utils/metrics.py::compute_ece`). The thesis used 15 bins; we
    keep that as the default for consistency.

    Args:
        probs:   (N, C) per-class predicted probabilities. Must sum to 1.0
                 along axis=1 (a softmax output, not raw logits).
        y_true:  (N,) integer ground-truth labels in [0, C).
        n_bins:  number of equal-width confidence bins.

    Returns:
        ECE as a float in [0, 1]. Returns NaN if `probs` is empty.

    Raises:
        ValueError: if `probs` does not look like a probability array.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D (N, C), got shape {probs.shape}")
    if len(probs) == 0:
        return float("nan")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
        raise ValueError(
            "probs rows do not sum to 1.0 — pass softmax outputs, not logits."
        )

    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for lower, upper in pairwise(bin_edges):
        # Right-closed bins so that perfect-confidence (1.0) lands in the last.
        in_bin = (confidences > lower) & (confidences <= upper)
        bin_size = in_bin.sum()
        if bin_size == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (bin_size / n) * abs(bin_conf - bin_acc)

    return float(ece)


def distribution_shift_drop(test_f1: float, eval_f1: float) -> float:
    """Relative drop in F1 from test → evaluation. Lower is better."""
    if test_f1 <= 0:
        return float("nan")
    return (test_f1 - eval_f1) / test_f1


def coverage(prediction_sets: np.ndarray, y_true: np.ndarray) -> float:
    """Empirical marginal coverage of conformal sets.

    Args:
        prediction_sets: (N, C) boolean array of class membership.
        y_true:          (N,) ground-truth labels.
    """
    return float(prediction_sets[np.arange(len(y_true)), y_true].mean())


def selective_risk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    deferred: np.ndarray,
) -> dict[str, float]:
    """Risk on the *non-deferred* subset plus the deferral rate.

    Returns:
        accepted_macro_f1: macro F1 over samples the system kept.
        deferral_rate:     fraction of samples deferred to a clinician.
        accepted_count:    number of accepted samples.
    """
    accepted = ~deferred
    if accepted.sum() == 0:
        return {
            "accepted_macro_f1": float("nan"),
            "deferral_rate": 1.0,
            "accepted_count": 0.0,
        }
    return {
        "accepted_macro_f1": macro_f1(y_true[accepted], y_pred[accepted]),
        "deferral_rate": float(deferred.mean()),
        "accepted_count": float(accepted.sum()),
    }
