"""BBSE label-shift correction — recovers a known synthetic shift."""

from __future__ import annotations

import numpy as np

from ps3c_robust.adapt import correct_probs, estimate_target_prior


def _sample(prior, cond, n, rng):
    """Draw n (true, pred) pairs: true ~ prior, pred ~ cond[:, true]."""
    y = rng.choice(len(prior), size=n, p=prior)
    yhat = np.array([rng.choice(cond.shape[0], p=cond[:, t]) for t in y])
    return y, yhat


def test_bbse_recovers_known_target_prior() -> None:
    rng = np.random.default_rng(0)
    p_source = np.array([0.50, 0.10, 0.40])
    q_target = np.array([0.30, 0.35, 0.35])          # unhealthy (class 1) up ~3.5x
    # column-stochastic P(pred | true): decent-but-imperfect predictor
    cond = np.array([
        [0.80, 0.15, 0.10],
        [0.10, 0.75, 0.10],
        [0.10, 0.10, 0.80],
    ])

    y_src, yhat_src = _sample(p_source, cond, 200_000, rng)
    _, yhat_tgt = _sample(q_target, cond, 200_000, rng)

    q_hat, w, C, nu, p_hat = estimate_target_prior(y_src, yhat_src, yhat_tgt, 3)

    assert np.allclose(q_hat.sum(), 1.0, atol=1e-9)
    # recovers the target prior within sampling error
    assert np.abs(q_hat - q_target).max() < 0.03, f"q_hat={q_hat} vs {q_target}"
    # recovers the unhealthy up-weight (w[1] = q/p ≈ 3.5)
    assert abs(w[1] - q_target[1] / p_source[1]) < 0.4


def test_correct_probs_renormalizes_and_identity() -> None:
    rng = np.random.default_rng(1)
    probs = rng.dirichlet([1, 1, 1], size=500).astype(np.float32)

    # uniform weights => unchanged (up to renormalization)
    same = correct_probs(probs, np.ones(3))
    assert np.allclose(same, probs, atol=1e-6)

    # arbitrary weights => rows still sum to 1, argmax can shift toward up-weighted class
    w = np.array([1.0, 5.0, 1.0])
    corrected = correct_probs(probs, w)
    assert np.allclose(corrected.sum(axis=1), 1.0, atol=1e-6)
    assert (corrected[:, 1] >= probs[:, 1] - 1e-6).mean() > 0.99
