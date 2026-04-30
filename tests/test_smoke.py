"""Smoke tests — these should pass before any weights/data arrive.

They exercise the parts of the pipeline that don't depend on Prof. Harangi's
inputs: the ensemble head, the conformal predictor, and the metrics.
"""

from __future__ import annotations

import numpy as np
import torch

from ps3c_robust.ensemble import AdaptiveEnsemble
from ps3c_robust.eval import (
    accuracy,
    coverage,
    distribution_shift_drop,
    expected_calibration_error,
    macro_f1,
    selective_risk,
)
from ps3c_robust.selective import ConformalPredictor


def test_adaptive_ensemble_shapes() -> None:
    model = AdaptiveEnsemble(num_teams=7, num_classes=4, hidden_dim=32)
    probs = torch.softmax(torch.randn(8, 7, 4), dim=-1)
    fused, weights = model(probs)
    assert fused.shape == (8, 4)
    assert weights.shape == (8, 7)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert torch.allclose(fused.sum(dim=-1), torch.ones(8), atol=1e-5)


def test_conformal_calibrate_predict_round_trip() -> None:
    rng = np.random.default_rng(0)
    n_cal, n_test, n_classes = 500, 100, 4

    cal_probs = _toy_softmax(rng, n_cal, n_classes)
    cal_labels = cal_probs.argmax(axis=1)        # cooperative oracle for smoke test
    test_probs = _toy_softmax(rng, n_test, n_classes)

    cp = ConformalPredictor(alpha=0.1, method="aps")
    cp.calibrate(cal_probs, cal_labels)
    out = cp.predict(test_probs)

    assert out.point_predictions.shape == (n_test,)
    assert out.prediction_sets.shape == (n_test, n_classes)
    assert out.prediction_sets.any(axis=1).all(), "every prediction set must be non-empty"
    assert out.deferred.shape == (n_test,)


def test_metrics() -> None:
    y_true = np.array([0, 1, 2, 3, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 0, 2])
    f1 = macro_f1(y_true, y_pred)
    assert 0.0 <= f1 <= 1.0

    acc = accuracy(y_true, y_pred)
    assert acc == 4 / 6

    drop = distribution_shift_drop(test_f1=0.95, eval_f1=0.85)
    assert 0.0 < drop < 0.2

    sets = np.array([
        [True, False, False, False],
        [False, True, False, False],
        [True, True, True, True],
        [False, False, True, False],
    ])
    cov = coverage(sets, np.array([0, 1, 2, 2]))
    assert cov == 1.0

    risk = selective_risk(
        y_true=np.array([0, 1, 2, 3]),
        y_pred=np.array([0, 1, 0, 0]),
        deferred=np.array([False, False, True, True]),
    )
    assert risk["deferral_rate"] == 0.5
    assert risk["accepted_count"] == 2.0


def test_expected_calibration_error() -> None:
    # Perfectly calibrated: 100% confidence, 100% correct → ECE = 0.
    perfect = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    labels = np.array([0, 1, 2])
    assert expected_calibration_error(perfect, labels) == 0.0

    # Severely miscalibrated: 100% confidence, 0% correct → ECE = 1.
    overconfident_wrong = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    wrong_labels = np.array([1, 2])
    assert expected_calibration_error(overconfident_wrong, wrong_labels) == 1.0

    # Reject raw logits (rows that don't sum to 1).
    logits = np.array([[2.0, 1.0, -1.0]])
    try:
        expected_calibration_error(logits, np.array([0]))
    except ValueError:
        pass
    else:
        raise AssertionError("ECE should reject non-probability inputs")


# --------------------------------------------------------------------- helpers


def _toy_softmax(rng: np.random.Generator, n: int, c: int) -> np.ndarray:
    logits = rng.normal(size=(n, c)).astype(np.float32)
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
