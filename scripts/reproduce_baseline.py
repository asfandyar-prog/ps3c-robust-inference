"""Reproduce the published per-team baselines against the official labels.

Reports accuracy and weighted-F1 for each team on both splits, alongside the
paper's reported figures, so the match is visible. With the official APACC
labels and the per-team decision rules in team_outputs.py (including DPZ's
two-stage cascade), every number matches the paper to four decimals.

Usage:
    python scripts/reproduce_baseline.py \
        --data-dir   /path/to/ps3c-team-data \
        --labels-dir /path/to/official-labels

The labels directory must contain:
    isbi2025-ps3c-test-dataset-annotated.csv
    isbi2025-ps3c-eval-dataset-annotated.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ps3c_robust.data.team_outputs import (
    TEAMS,
    load_official_labels,
    team_predictions,
)

# Published figures from the PS3C paper (Tables 3 and 4).
PAPER_ACC = {
    "test": {"YMG": 0.8686, "JNG": 0.8700, "NGU": 0.8499, "CHA": 0.8165, "GUP": 0.8604, "DPZ": 0.8627, "WAN": 0.7901},
    "eval": {"YMG": 0.7996, "JNG": 0.8229, "NGU": 0.7581, "CHA": 0.7953, "GUP": 0.7713, "DPZ": 0.7687, "WAN": 0.7723},
}
PAPER_F1 = {
    "test": {"YMG": 0.8680, "JNG": 0.8702, "NGU": 0.8523, "CHA": 0.8319, "GUP": 0.8604, "DPZ": 0.8622, "WAN": 0.8058},
    "eval": {"YMG": 0.7858, "JNG": 0.8176, "NGU": 0.7136, "CHA": 0.7944, "GUP": 0.7211, "DPZ": 0.7092, "WAN": 0.7615},
}


def evaluate(data_dir: Path, labels_dir: Path, split: str) -> None:
    gt = load_official_labels(labels_dir, split)
    preds = {t: team_predictions(data_dir, t, split) for t in TEAMS}

    common = set(gt)
    for t in TEAMS:
        common &= set(preds[t])
    names = sorted(common)
    y = np.array([gt[n] for n in names])

    print(f"===== {split}  ({len(names)} images) =====")
    print(f"{'team':<5}{'acc':>9}{'paper':>9}{'wF1':>9}{'paper':>9}")
    for t in TEAMS:
        p = np.array([preds[t][n] for n in names])
        acc = accuracy_score(y, p)
        wf1 = f1_score(y, p, average="weighted")
        print(
            f"{t:<5}{acc:>9.4f}{PAPER_ACC[split][t]:>9.4f}"
            f"{wf1:>9.4f}{PAPER_F1[split][t]:>9.4f}"
        )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    args = parser.parse_args()

    for split in ("test", "eval"):
        evaluate(args.data_dir, args.labels_dir, split)

    print(
        "All teams should match the paper to four decimals on both accuracy and\n"
        "weighted-F1. Note: the paper's 'F1-score' column is weighted F1, not\n"
        "macro F1, despite the text describing macro F1."
    )


if __name__ == "__main__":
    main()
