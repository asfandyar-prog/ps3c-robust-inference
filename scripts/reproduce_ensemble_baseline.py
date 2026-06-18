"""Reproduce all eight ensemble strategies from the PS3C paper (all 7 teams).

Trains each ensemble on the test-set predictions and evaluates on the
evaluation set, matching the paper's setup (Section 3.8, Tables 5 and 6).

The eight strategies:
    1.  Simple Average
    2.  Hard Voting
    3.  Weighted Average      (weights optimised on the test set)
    4.  Rank Averaging
    5.  Geometric Mean
    6.  Stacking — Logistic Regression
    7.  Stacking — Random Forest
    8.  Stacking — Gradient Boost    <- paper: 0.9245 eval weighted-F1

Usage:
    python scripts/reproduce_ensemble_baseline.py \
        --data-dir   E:/ps3c/ps3c-team-data \
        --labels-dir E:/ps3c/ps3c-labels

Published figures (paper Tables 5 and 6, weighted-F1):
    Test:  SA 0.8686 | HV 0.8710 | WA 0.8723 | RA 0.7879 | GM 0.8633
           LR 0.8741 | RF 0.9250 | GB 0.9517
    Eval:  SA 0.7669 | HV 0.7817 | WA 0.8117 | RA 0.8250 | GM 0.7331
           LR 0.8497 | RF 0.8989 | GB 0.9245
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from ps3c_robust.data.team_outputs import (
    CANON_IDX,
    DPZ_RUBBISH_THRESHOLD,
    HEADER_ALIASES,
    TEAMS,
    _normalize_name,
    _team_file_map,
    load_official_labels,
)

PAPER = {
    "test": {
        "Simple Average": 0.8686, "Hard Voting": 0.8710, "Weighted Average": 0.8723,
        "Rank Averaging": 0.7879, "Geometric Mean": 0.8633,
        "Stacking LR": 0.8741, "Stacking RF": 0.9250, "Stacking GB": 0.9517,
    },
    "eval": {
        "Simple Average": 0.7669, "Hard Voting": 0.7817, "Weighted Average": 0.8117,
        "Rank Averaging": 0.8250, "Geometric Mean": 0.7331,
        "Stacking LR": 0.8497, "Stacking RF": 0.8989, "Stacking GB": 0.9245,
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_probs(data_dir: Path, labels_dir: Path, split: str):
    """Return (probs, labels): (N, T, C) float64 and (N,) int64."""
    import csv
    gt = load_official_labels(labels_dir, split)
    files = _team_file_map(data_dir)
    per_team: dict[str, dict[str, np.ndarray]] = {}

    for team in TEAMS:
        path = files[team][split]
        with open(path, newline="") as f:
            rd = csv.reader(f)
            header = [h.strip().lower() for h in next(rd)]
            rows = [r for r in rd if r and r[0].strip()]
        d: dict[str, np.ndarray] = {}
        if team == "DPZ":
            ri, hi, ui = header.index("rubbish"), header.index("healthy"), header.index("unhealthy")
            for row in rows:
                nm = _normalize_name(row[0])
                rub, hea, unh = float(row[ri]), float(row[hi]), float(row[ui])
                v = np.array([0.0, 0.0, rub]) if rub >= DPZ_RUBBISH_THRESHOLD else np.array([hea, unh, 1.0 - rub])
                s = v.sum(); d[nm] = v / s if s > 0 else v
        else:
            c2c = {i: HEADER_ALIASES[c] for i, c in enumerate(header) if c in HEADER_ALIASES}
            for row in rows:
                nm = _normalize_name(row[0]); v = np.zeros(3)
                for i, cls in c2c.items():
                    if cls == "bothcells": continue
                    try: v[CANON_IDX[cls]] = float(row[i])
                    except (ValueError, IndexError): pass
                s = v.sum(); d[nm] = v / s if s > 0 else v
        per_team[team] = d

    common = set(gt)
    for t in TEAMS:
        common &= set(per_team[t])
    names = sorted(common)
    probs = np.stack([np.array([per_team[t][n] for n in names]) for t in TEAMS], axis=1)
    labels = np.array([gt[n] for n in names])
    return probs, labels


# ---------------------------------------------------------------------------
# Eight ensemble strategies
# ---------------------------------------------------------------------------

def simple_average(p_tr, y_tr, p_te):
    return p_te.mean(axis=1)


def hard_voting(p_tr, y_tr, p_te):
    N, T = p_te.shape[:2]
    votes = p_te.argmax(axis=2)
    out = np.zeros((N, 3))
    for i in range(N):
        for t in range(T):
            out[i, votes[i, t]] += 1
    return out / T


def weighted_average(p_tr, y_tr, p_te):
    T = p_tr.shape[1]
    def neg_f1(w):
        w = np.abs(w) / np.abs(w).sum()
        fused = (p_tr * w[None, :, None]).sum(axis=1)
        return -f1_score(y_tr, fused.argmax(axis=1), average="weighted")
    res = minimize(neg_f1, np.ones(T) / T, method="Nelder-Mead",
                   options={"maxiter": 2000, "xatol": 1e-5})
    w = np.abs(res.x) / np.abs(res.x).sum()
    return (p_te * w[None, :, None]).sum(axis=1)


def rank_averaging(p_tr, y_tr, p_te):
    N, T, C = p_te.shape
    ranks = np.zeros_like(p_te)
    for t in range(T):
        for c in range(C):
            order = np.argsort(p_te[:, t, c])
            ranks[order, t, c] = np.arange(N)
    return ranks.mean(axis=1)


def geometric_mean(p_tr, y_tr, p_te):
    geo = np.exp(np.log(np.clip(p_te, 1e-12, None)).mean(axis=1))
    return geo / geo.sum(axis=1, keepdims=True)


STACKING_CLFS = {
    "Stacking LR": LogisticRegression(max_iter=1000, random_state=42),
    "Stacking RF": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "Stacking GB": GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                               learning_rate=0.05, random_state=42),
}

STRATEGIES = [
    ("Simple Average",   simple_average),
    ("Hard Voting",      hard_voting),
    ("Weighted Average", weighted_average),
    ("Rank Averaging",   rank_averaging),
    ("Geometric Mean",   geometric_mean),
    ("Stacking LR",      None),
    ("Stacking RF",      None),
    ("Stacking GB",      None),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",   type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    args = parser.parse_args()

    print("Loading test split (used to train meta-learners)...")
    p_tr, y_tr = load_probs(args.data_dir, args.labels_dir, "test")
    N, T, C = p_tr.shape
    print(f"  {N} images, {T} teams\n")

    print("Loading eval split (used for evaluation)...")
    p_ev, y_ev = load_probs(args.data_dir, args.labels_dir, "eval")
    print(f"  {len(y_ev)} images\n")

    X_tr = p_tr.reshape(N, T * C)
    X_ev = p_ev.reshape(len(y_ev), T * C)

    w = 22
    print(f"{'Strategy':<{w}} {'Test wF1':>10} {'Paper':>8} {'Eval wF1':>10} {'Paper':>8}")
    print("-" * (w + 40))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, fn in STRATEGIES:
        print(f"  running {name}...", end="\r", flush=True)

        if fn is not None:
            # Non-stacking: no fitting needed
            f1_tr = f1_score(y_tr, fn(p_tr, y_tr, p_tr).argmax(axis=1), average="weighted")
            f1_ev = f1_score(y_ev, fn(p_tr, y_tr, p_ev).argmax(axis=1), average="weighted")
        else:
            # Stacking: cross-val on test for honest test-set score, full fit for eval
            clf = STACKING_CLFS[name]
            oof = np.zeros(N, dtype=int)
            for fold_tr, fold_val in skf.split(X_tr, y_tr):
                m = clone(clf)
                m.fit(X_tr[fold_tr], y_tr[fold_tr])
                oof[fold_val] = m.predict(X_tr[fold_val])
            f1_tr = f1_score(y_tr, oof, average="weighted")
            clf_full = clone(clf)
            clf_full.fit(X_tr, y_tr)
            f1_ev = f1_score(y_ev, clf_full.predict(X_ev), average="weighted")

        p_tr_ref = PAPER["test"].get(name, float("nan"))
        p_ev_ref = PAPER["eval"].get(name, float("nan"))
        print(f"{name:<{w}} {f1_tr:>10.4f} {p_tr_ref:>8.4f} {f1_ev:>10.4f} {p_ev_ref:>8.4f}")

    print()
    print("Test column uses 5-fold CV for stacking methods (honest estimate).")
    print("Eval column uses meta-learner trained on full test set.")
    print("The Gradient Boost eval score is the baseline this project targets to beat.")


if __name__ == "__main__":
    main()
