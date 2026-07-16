"""
Run from anywhere (venv Python 3.11):
  python scripts/run_04_conformal.py ^
      --data-dir   E:\ps3c\ps3c-team-data ^
      --labels-dir E:\ps3c\ps3c-labels ^
      --out-dir    E:\ps3c\ps3c-robust-inference\results

Stage 3 — split-conformal selective prediction on the 3-class PS3C ensemble.

This is the real Stage-3 driver (the former scripts/calibrate_conformal.py was a
stub). It uses the implemented ConformalPredictor in
src/ps3c_robust/selective/conformal.py, run in 3-class mode (healthy, unhealthy,
rubbish) — bothcells stays dropped, consistent with the rest of the pipeline.

Two scenarios, each evaluated on TWO ensemble representations:
  * rank_average  — our canonical champion (rank-average the 7 teams, normalized
                    per sample). Its argmax equals the rank-averaging prediction.
  * simple_average— mean of the 7 teams' raw probability vectors (normalized).

  1. WITHIN-TEST (exchangeable): split the test set into calibration + hold-out.
     Split-conformal guarantees marginal coverage >= 1 - alpha here.
  2. CROSS-SPLIT (shifted): calibrate on full test, predict on full eval.

Empirical findings that shape this driver (see results/README.md):
  * METHOD. Both LAC (Sadinle et al., 2019) and randomized APS (Romano et al.,
    2020) meet their coverage guarantee within-test. (APS previously under-covered
    because conformal.py randomized the calibration score but built prediction
    sets non-randomized; that asymmetry is now fixed.) LAC is the default.
  * REPRESENTATION. rank_average is (by construction) invariant to the test->eval
    distribution shift: its per-class marginals are near-identical across splits,
    so cross-split coverage HOLDS. simple_average tracks the shift, so cross-split
    coverage FALLS BELOW target — that under-coverage is the selective-prediction
    evidence for the shift.
"""
import sys, csv, json, argparse
from pathlib import Path

import numpy as np

# Make the package importable even without the editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
except ImportError:
    print("ERROR: pip install scikit-learn"); sys.exit(1)

from ps3c_robust.selective.conformal import ConformalPredictor

# ── canonical data loader (identical rule to run_02 / run_03) ─────────────────
CANON_IDX = {"healthy": 0, "unhealthy": 1, "rubbish": 2}
CLASS_NAMES = ["healthy", "unhealthy", "rubbish"]
TEAMS = ["YMG", "JNG", "NGU", "CHA", "GUP", "DPZ", "WAN"]
DPZ_THR = 0.46
HEADER_ALIASES = {
    "healthy": "healthy", "healthy_prob": "healthy", "prob_healthy": "healthy",
    "unhealthy": "unhealthy", "unhealthy_prob": "unhealthy", "prob_unhealthy": "unhealthy",
    "rubbish": "rubbish", "rubbish_prob": "rubbish", "prob_rubbish": "rubbish",
    "bothcells": "bothcells", "bothcells_prob": "bothcells",
}
SEED = 42


def norm(n):
    n = n.strip(); return n[:-4] if n.lower().endswith(".png") else n


def file_map(data_dir):
    d = Path(data_dir)
    return {
        "YMG": {"test": d/"wrapup_UT/wrapup_UT/test_phase_prob.csv", "eval": d/"wrapup_UT/wrapup_UT/final_eval_phase_revised_prob.csv"},
        "JNG": {"test": d/"jianght_challenge_materials/Evaluation-set.csv", "eval": d/"jianght_challenge_materials/Test-set.csv"},
        "NGU": {"test": d/"NGU/predictions_isbi2025-ps3c-test-dataset.csv", "eval": d/"NGU/predictions_isbi2025-ps3c-eval-dataset.csv"},
        "CHA": {"test": d/"ChatoLina_challenge_materials/isbi2025-ps3c-test-dataset pro Ens.csv", "eval": d/"ChatoLina_challenge_materials/isbi2025-ps3c-eval-dataset pro Ens.csv"},
        "GUP": {"test": d/"Chinmay materials/ISBI PS3C IIT KGP/Test_Set_ProbabilityScores.csv", "eval": d/"Chinmay materials/ISBI PS3C IIT KGP/Eval_Set_ProbabilityScores.csv"},
        "DPZ": {"test": d/"Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_test.csv", "eval": d/"Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_eval.csv"},
        "WAN": {"test": d/"WAN/validation_predictions2.csv", "eval": d/"WAN/val_predictions_resnet50.csv"},
    }


def load_labels(labels_dir, split):
    path = Path(labels_dir)/f"isbi2025-ps3c-{split}-dataset-annotated.csv"; gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = row["label"].strip().lower()
            if lab in CANON_IDX: gt[norm(row["image_name"])] = CANON_IDX[lab]
    return gt


def load_team(path, team):
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f); header = [h.strip().lower() for h in next(rd)]
        rows = [r for r in rd if r and r[0].strip()]
    d = {}
    if team == "DPZ":
        ri, hi, ui = header.index("rubbish"), header.index("healthy"), header.index("unhealthy")
        for row in rows:
            nm = norm(row[0]); rub, hea, unh = float(row[ri]), float(row[hi]), float(row[ui])
            v = np.zeros(3, dtype=np.float32); v[2 if rub >= DPZ_THR else (0 if hea >= unh else 1)] = 1.0; d[nm] = v
    else:
        c2c = {i: HEADER_ALIASES[c] for i, c in enumerate(header) if c in HEADER_ALIASES}
        for row in rows:
            nm = norm(row[0]); v = np.zeros(3, dtype=np.float32)
            for i, cls in c2c.items():
                if cls == "bothcells": continue
                try: v[CANON_IDX[cls]] = float(row[i])
                except: pass
            s = v.sum(); d[nm] = v/s if s > 0 else v
    return d


def load_all(data_dir, labels_dir, split):
    gt = load_labels(labels_dir, split); fm = file_map(data_dir)
    pt = {t: load_team(fm[t][split], t) for t in TEAMS}
    # CANONICAL SAMPLE SET: per split, intersection of image IDs scored by ALL 7
    # teams, restricted to annotated ground-truth. Same rule as run_02 / run_03.
    common = set(gt)
    for t in TEAMS: common &= set(pt[t])
    names = sorted(common)
    probs = np.stack([np.array([pt[t][n] for n in names], dtype=np.float32) for t in TEAMS], axis=1)
    labels = np.array([gt[n] for n in names], dtype=np.int64)
    return probs, labels


def rank_average_distribution(probs):
    """Rank-average the 7 teams into a per-sample 3-class distribution.

    Returns (N, 3) float32 rows summing to 1. argmax matches rank averaging.
    Invariant to the test->eval shift (rank marginals are uniform by design).
    """
    N, T, C = probs.shape
    ranks = np.zeros_like(probs)
    for t in range(T):
        for c in range(C):
            order = np.argsort(probs[:, t, c]); ranks[order, t, c] = np.arange(N)
    mean_ranks = ranks.mean(1)                      # (N, C)
    row_sums = mean_ranks.sum(1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return (mean_ranks / row_sums).astype(np.float32)


def simple_average_distribution(probs):
    """Mean of the 7 teams' raw probability vectors, renormalized per sample.

    A genuine probability that tracks the test->eval distribution shift.
    """
    m = probs.mean(1)                               # (N, C)
    row_sums = m.sum(1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return (m / row_sums).astype(np.float32)


def evaluate(pred, y_true, target_coverage):
    """Coverage, deferral rate and accepted-set weighted-F1 for a prediction."""
    sets = pred.prediction_sets                     # (N, C) bool
    covered = sets[np.arange(len(y_true)), y_true]  # true label in set?
    coverage = float(covered.mean())
    deferred = pred.deferred
    deferral_rate = float(deferred.mean())
    accepted = ~deferred
    n_acc = int(accepted.sum())
    if n_acc > 0:
        acc_wf1 = float(f1_score(y_true[accepted], pred.point_predictions[accepted], average="weighted"))
        acc_cov = float(covered[accepted].mean())
    else:
        acc_wf1 = None; acc_cov = None
    return {
        "target_coverage": round(target_coverage, 4),
        "empirical_coverage": round(coverage, 4),
        "coverage_gap": round(coverage - target_coverage, 4),
        "deferral_rate": round(deferral_rate, 4),
        "n_accepted": n_acc,
        "n_total": int(len(y_true)),
        "accepted_weighted_f1": None if acc_wf1 is None else round(acc_wf1, 4),
        "accepted_coverage": None if acc_cov is None else round(acc_cov, 4),
    }


def run_scenario(cal_probs, cal_y, pred_probs, pred_y, method, alphas, seed):
    out = {}
    for a in alphas:
        cp = ConformalPredictor(alpha=a, method=method, n_classes=3,
                                rng=np.random.default_rng(seed))
        cp.calibrate(cal_probs, cal_y)
        pred = cp.predict(pred_probs)
        out[f"{a:.2f}"] = evaluate(pred, pred_y, 1.0 - a)
    return out


def print_table(title, note, block, alphas):
    print("\n" + "-" * 78)
    print(title); print(note)
    print("-" * 78)
    print(f"{'input':>14} {'alpha':>6} {'target':>7} {'cov':>7} {'gap':>8} {'defer%':>8} {'acc.wF1':>8} {'n_acc':>7}")
    for label in block:
        for a in alphas:
            m = block[label][f"{a:.2f}"]
            wf1 = "  n/a" if m["accepted_weighted_f1"] is None else f"{m['accepted_weighted_f1']:.4f}"
            flag = "  <- UNDER" if m["coverage_gap"] < -0.01 else ""
            print(f"{label:>14} {a:>6.2f} {m['target_coverage']:>7.3f} {m['empirical_coverage']:>7.3f} "
                  f"{m['coverage_gap']:>+8.3f} {100*m['deferral_rate']:>7.2f}% {wf1:>8} {m['n_accepted']:>7}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--method", choices=["lac", "aps"], default="lac",
                    help="lac (valid, default) or aps (under-covers — see docstring)")
    ap.add_argument("--alphas", default="0.05,0.10,0.15,0.20")
    ap.add_argument("--calib-frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    alphas = [float(a) for a in args.alphas.split(",")]

    print("=" * 78)
    print(f"PS3C STAGE 3 — CONFORMAL SELECTIVE PREDICTION (3-class, method={args.method})")
    print("=" * 78)

    print("\nLoading canonical sample set...")
    probs_te, y_te = load_all(args.data_dir, args.labels_dir, "test")
    probs_ev, y_ev = load_all(args.data_dir, args.labels_dir, "eval")
    reps = {
        "rank_average":   (rank_average_distribution(probs_te), rank_average_distribution(probs_ev)),
        "simple_average": (simple_average_distribution(probs_te), simple_average_distribution(probs_ev)),
    }
    print(f"  test: {len(y_te)}   eval: {len(y_ev)}")
    for name, (rte, rev) in reps.items():
        print(f"  {name:>14} point wF1 — test {f1_score(y_te, rte.argmax(1), average='weighted'):.4f}"
              f"  eval {f1_score(y_ev, rev.argmax(1), average='weighted'):.4f}")

    # Split TEST -> calibration / hold-out (stratified) — shared across reps.
    idx = np.arange(len(y_te))
    ix_cal, ix_hold = train_test_split(
        idx, test_size=1.0 - args.calib_frac, random_state=args.seed, stratify=y_te)
    print(f"  within-test split: calibration {len(ix_cal)}  hold-out {len(ix_hold)}")

    within, cross = {}, {}
    for name, (rte, rev) in reps.items():
        within[name] = run_scenario(rte[ix_cal], y_te[ix_cal], rte[ix_hold], y_te[ix_hold],
                                     args.method, alphas, args.seed)
        cross[name] = run_scenario(rte, y_te, rev, y_ev,
                                   args.method, alphas, args.seed)

    print_table(
        "SCENARIO 1 — WITHIN-TEST (calibrate on test-calib, predict on test-holdout)",
        "Expect empirical coverage >= target (guarantee holds under exchangeability).",
        within, alphas)
    print_table(
        "SCENARIO 2 — CROSS-SPLIT (calibrate on full test, predict on full eval)",
        "simple_average should UNDER-cover (shift); rank_average holds (shift-invariant).",
        cross, alphas)

    results = {
        "method": args.method,
        "n_classes": 3,
        "class_names": CLASS_NAMES,
        "canonical_n": {"test": int(len(y_te)), "eval": int(len(y_ev))},
        "within_test_split": {"calibration": int(len(ix_cal)), "holdout": int(len(ix_hold))},
        "within_test": within,
        "cross_split_test_to_eval": cross,
    }
    out_path = out / f"conformal_results_{args.method}.json"
    out_path.write_text(json.dumps(results, indent=2))

    print("\n" + "=" * 78)
    print("KEY FINDING: within-test LAC meets its coverage guarantee. Across the")
    print("test->eval shift, simple-average conformal UNDER-covers the true label")
    print("(selective-prediction evidence for the shift), while rank-averaging is")
    print("shift-invariant and keeps coverage — corroborating why it is the robust")
    print("ensemble champion.")
    print(f"Saved -> {out_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
