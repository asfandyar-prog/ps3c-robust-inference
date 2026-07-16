"""Conformal prediction with selective deferral for the PS3C ensemble.

Implements split-conformal classification with two scoring options:

* `lac` — Least Ambiguous Set (Sadinle et al., 2019)
* `aps` — Adaptive Prediction Sets (Romano et al., 2020); recommended default.

Coverage guarantee (under exchangeability): with calibration set of size n and
nominal level α, the marginal coverage of the prediction sets on a fresh point
is at least 1 − α.

Deferral logic on top of the prediction set:
    * If the set has more than one label → defer to cytologist.
    * If the set contains "bothcells" with probability above a threshold → defer.

Bothcells are *expected* to cluster in the deferred region — the original PS3C
teams ignored bothcells entirely, so any system prediction on them is
unreliable. Stage 3 documents this empirically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ScoreMethod = Literal["lac", "aps"]

# Class index convention — keep in sync with data loader.
CLASS_NAMES = ["healthy", "unhealthy", "rubbish", "bothcells"]
BOTHCELLS_IDX = CLASS_NAMES.index("bothcells")


@dataclass
class SelectivePrediction:
    """Output of `ConformalPredictor.predict`."""

    point_predictions: np.ndarray   # (N,) argmax labels (only meaningful where deferred=False)
    prediction_sets: np.ndarray     # (N, C) boolean mask of class membership in the conformal set
    deferred: np.ndarray            # (N,) boolean — True means defer to clinician
    scores: np.ndarray              # (N,) maximum nonconformity score (debug / coverage plots)


class ConformalPredictor:
    """Split-conformal predictor with deferral."""

    def __init__(
        self,
        alpha: float = 0.05,
        method: ScoreMethod = "aps",
        defer_on_set_size_gt: int = 1,
        bothcells_in_set_threshold: float = 0.10,
        n_classes: int = 3,
        rng: np.random.Generator | None = None,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.method = method
        self.defer_on_set_size_gt = defer_on_set_size_gt
        self.bothcells_in_set_threshold = bothcells_in_set_threshold
        self.n_classes = n_classes
        # The bothcells deferral branch only applies when the data actually
        # carries a bothcells column (4-class layout). The PS3C pipeline runs
        # 3-class (healthy, unhealthy, rubbish) with bothcells dropped, so with
        # the default n_classes=3 the branch is disabled and deferral is driven
        # purely by prediction-set size.
        self._defer_on_bothcells = n_classes > BOTHCELLS_IDX
        self.rng = rng or np.random.default_rng(0)

        self._qhat: float | None = None

    def calibrate(self, probs: np.ndarray, labels: np.ndarray) -> None:
        """Compute the conformal quantile q-hat from a calibration split."""
        if probs.shape[1] != self.n_classes:
            raise ValueError(
                f"probs has {probs.shape[1]} columns but n_classes={self.n_classes}"
            )
        scores = self._nonconformity(probs, labels)
        n = len(scores)
        # finite-sample correction
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        self._qhat = float(np.quantile(scores, q_level, method="higher"))

    def predict(self, probs: np.ndarray) -> SelectivePrediction:
        """Produce conformal sets and deferral decisions for new data."""
        if self._qhat is None:
            raise RuntimeError("Call calibrate() before predict().")
        if probs.shape[1] != self.n_classes:
            raise ValueError(
                f"probs has {probs.shape[1]} columns but n_classes={self.n_classes}"
            )

        sets = self._build_sets(probs, self._qhat)             # (N, C) bool
        point = probs.argmax(axis=1)                           # (N,)
        set_size = sets.sum(axis=1)                            # (N,)

        deferred = set_size > self.defer_on_set_size_gt
        if self._defer_on_bothcells:
            bothcells_in_set = sets[:, BOTHCELLS_IDX] & (
                probs[:, BOTHCELLS_IDX] > self.bothcells_in_set_threshold
            )
            deferred = deferred | bothcells_in_set

        # Per-sample worst-case score is informative for coverage plots.
        scores = self._nonconformity_max(probs)

        return SelectivePrediction(
            point_predictions=point,
            prediction_sets=sets,
            deferred=deferred,
            scores=scores,
        )

    # ----------------------------------------------------------------- scoring

    def _nonconformity(self, probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
        if self.method == "lac":
            return 1.0 - probs[np.arange(len(labels)), labels]

        # APS: cumulative mass of classes ranked from most to least likely
        # up to and including the true class.
        order = np.argsort(-probs, axis=1)
        ranks = np.empty_like(order)
        rows = np.arange(probs.shape[0])[:, None]
        ranks[rows, order] = np.arange(probs.shape[1])
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cumulative = np.cumsum(sorted_probs, axis=1)
        true_rank = ranks[np.arange(len(labels)), labels]
        scores = cumulative[np.arange(len(labels)), true_rank]
        # Optional randomization removes conservativeness bias.
        u = self.rng.uniform(size=len(labels))
        scores -= u * sorted_probs[np.arange(len(labels)), true_rank]
        return scores

    def _nonconformity_max(self, probs: np.ndarray) -> np.ndarray:
        if self.method == "lac":
            return 1.0 - probs.max(axis=1)
        # For APS the "score" in monitoring is just 1 - max_prob.
        return 1.0 - probs.max(axis=1)

    def _build_sets(self, probs: np.ndarray, qhat: float) -> np.ndarray:
        if self.method == "lac":
            return probs >= 1.0 - qhat

        # APS set construction: include classes in descending-probability order
        # while cumulative mass ≤ qhat (and always include the top class).
        order = np.argsort(-probs, axis=1)
        sorted_probs = np.take_along_axis(probs, order, axis=1)
        cumulative = np.cumsum(sorted_probs, axis=1)
        include_sorted = cumulative <= qhat
        include_sorted[:, 0] = True   # never empty

        sets = np.zeros_like(probs, dtype=bool)
        rows = np.arange(probs.shape[0])[:, None]
        sets[rows, order] = include_sorted
        return sets
