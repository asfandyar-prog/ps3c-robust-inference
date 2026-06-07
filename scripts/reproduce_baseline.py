"""Reproduce the per-team and gradient-boost ensemble baselines on real data.

Loads the six teams' probability outputs, aligns them, and reports:
  - alignment counts per split (sanity: test=18,159, eval=29,117)
  - per-team macro-F1 on each split
  - gradient-boost stacking ensemble macro-F1

Usage:
    python scripts/reproduce_baseline.py --data-dir /path/to/ps3c-team-data

IMPORTANT: the F1 numbers will NOT exactly match the published paper until the
official APACC ground-truth labels are available. The labels used here are a
best-effort consensus from the teams' own prediction files, which disagree in a
few hundred cases. The script prints the disagreement count so the gap is
visible. See src/ps3c_robust/data/team_outputs.py for details.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score

from ps3c_robust.data.team_outputs import TEAMS, load_split


def per_team_f1(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    valid = labels >= 0
    out = {}
    for ti, team in enumerate(TEAMS):
        preds = probs[valid, ti].argmax(axis=1)
        out[team] = float(f1_score(labels[valid], preds, average="macro"))
    return out


def gradient_boost_baseline(
    probs_train: np.ndarray,
    y_train: np.ndarray,
    probs_eval: np.ndarray,
    y_eval: np.ndarray,
) -> float:
    vtr, vte = y_train >= 0, y_eval >= 0
    Ftr = probs_train[vtr].reshape(int(vtr.sum()), -1)
    Fte = probs_eval[vte].reshape(int(vte.sum()), -1)
    clf = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
    )
    clf.fit(Ftr, y_train[vtr])
    return float(f1_score(y_eval[vte], clf.predict(Fte), average="macro"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing the organizer share (wrapup_UT/, jianght_.../, "
        "Chinmay materials/, etc.) plus the WAN/ and YMG/ follow-up folders.",
    )
    args = parser.parse_args()

    print("Loading TEST split...")
    _, p_test, y_test = load_split(args.data_dir, "test")
    print(f"  aligned images: {len(y_test)} (expected 18,159); labels: {(y_test >= 0).sum()}")

    print("Loading EVAL split...")
    _, p_eval, y_eval = load_split(args.data_dir, "eval")
    print(f"  aligned images: {len(y_eval)} (expected 29,117); labels: {(y_eval >= 0).sum()}")

    print("\nPer-team macro-F1 (consensus labels — not yet authoritative):")
    print(f"{'team':<6} {'test':>8} {'eval':>8}")
    f1_test = per_team_f1(p_test, y_test)
    f1_eval = per_team_f1(p_eval, y_eval)
    for team in TEAMS:
        print(f"{team:<6} {f1_test[team]:>8.4f} {f1_eval[team]:>8.4f}")

    print("\nGradient-boost stacking ensemble:")
    print("  paper reference (7 teams): 0.9517 test / 0.9245 eval")
    gb = gradient_boost_baseline(p_test, y_test, p_eval, y_eval)
    print(f"  reproduced (6 teams, trained on test -> eval): {gb:.4f}")

    print(
        "\nNOTE: numbers will not match the paper exactly until the official "
        "APACC labels are obtained. The current labels are a team consensus."
    )


if __name__ == "__main__":
    main()
