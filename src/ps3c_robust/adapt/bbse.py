"""Black Box Shift Estimation (BBSE) for label-shift correction.

Reference
---------
Lipton, Wang, Smola. "Detecting and Correcting for Label Shift with Black Box
Predictors." ICML 2018.

Assumption
----------
Label shift: the class-conditional p(x | y) is identical in source and target;
only the label prior p(y) changes. Under this assumption, for a fixed predictor
f the importance weights

    w(y) = q_target(y) / p_source(y)

satisfy the linear system

    nu = C w

where
    C[i, j] = P_source(f(x) = i, y = j)      # joint confusion matrix
    nu[i]   = P_target(f(x) = i)             # target predicted-label distribution

We solve it by least squares (this is BBSE-*hard*: the confusion matrix is built
from hard argmax predictions), clip negative weights to zero, and recover the
target prior q(y) = w(y) * p_source(y), renormalized to sum to one.

Notes / caveats
---------------
* Any class whose column of C is poorly estimated (few source samples, or a
  predictor that rarely gets that class right) yields a noisy/ill-conditioned
  weight. Check the conditioning of C before trusting a per-class weight.
* correct_probs applies the standard label-shift posterior correction
  p_target(y | x) ∝ p_source(y | x) * w(y).
"""

from __future__ import annotations

import numpy as np


def confusion_matrix_joint(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Joint confusion matrix C[i, j] = P(pred = i, true = j) on the source set."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"length mismatch: {len(y_true)} vs {len(y_pred)}")
    C = np.zeros((n_classes, n_classes), dtype=np.float64)
    np.add.at(C, (y_pred, y_true), 1.0)   # rows = predicted, cols = true
    return C / len(y_true)


def estimate_shift_weights(
    y_true_source: np.ndarray,
    y_pred_source: np.ndarray,
    y_pred_target: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """BBSE-hard importance weights w[j] = q_target(j) / p_source(j), clipped >= 0.

    Returns (w, C, nu): the weights, the joint confusion matrix, and the target
    predicted-label distribution.
    """
    C = confusion_matrix_joint(y_true_source, y_pred_source, n_classes)
    y_pred_target = np.asarray(y_pred_target)
    nu = np.bincount(y_pred_target, minlength=n_classes).astype(np.float64) / len(y_pred_target)
    w, *_ = np.linalg.lstsq(C, nu, rcond=None)
    w = np.clip(w, 0.0, None)
    return w, C, nu


def estimate_target_prior(
    y_true_source: np.ndarray,
    y_pred_source: np.ndarray,
    y_pred_target: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate the target class prior q(y) via BBSE.

    Returns (q, w, C, nu, p_source) where q is renormalized to sum to one. If the
    clipped weights zero out every class, falls back to the source prior.
    """
    w, C, nu = estimate_shift_weights(y_true_source, y_pred_source, y_pred_target, n_classes)
    p_source = np.bincount(np.asarray(y_true_source), minlength=n_classes).astype(np.float64)
    p_source /= p_source.sum()
    q = w * p_source
    s = q.sum()
    q = q / s if s > 0 else p_source.copy()
    return q, w, C, nu, p_source


def correct_probs(probs: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Label-shift posterior correction: p_target(y|x) ∝ p_source(y|x) * w(y).

    Args:
        probs: (N, C) source posterior probabilities.
        w:     (C,) importance weights from BBSE.
    Returns:
        (N, C) reweighted, per-row renormalized probabilities.
    """
    probs = np.asarray(probs)
    c = probs * np.asarray(w, dtype=probs.dtype)[None, :]
    s = c.sum(axis=1, keepdims=True)
    return (c / np.where(s == 0, 1.0, s)).astype(probs.dtype)
