"""Metrics used across all three stages.

The most important number for the paper is `distribution_shift_drop`: the
test→eval F1 collapse that the original challenge reported and that this work
sets out to mitigate.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-averaged F1 — the headline number from the original paper."""
    return float(f1_score(y_true, y_pred, average="macro"))


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
