"""
Run (venv Python 3.11):
  python scripts/run_07_conformal_analysis.py ^
      --data-dir   E:\ps3c\ps3c-team-data ^
      --labels-dir E:\ps3c\ps3c-labels ^
      --out-dir    E:\ps3c\ps3c-robust-inference\results

Stage 3 — deep analysis of conformal selective prediction (LAC, 3-class,
canonical loader identical to run_04_conformal). No model is retrained.

Two ensembles are analysed on equal footing:
  * rank_average  — the shift-invariant champion (eval point wF1 0.8157)
  * simple_average— the ensemble the existing conformal numbers used

Experiments:
  1. Shift attribution — is the cross-split coverage failure due to shift?
     Compare, on the SAME eval hold-out, TEST-calibrated (cross-split) vs
     EVAL-calibrated (within-eval, exchangeable) coverage. Also reproduce the
     existing test-calibrated full-eval numbers (discrepancy check).
  2. Deferred set (deployment = test-calibrated on full eval): set-size
     distribution, class composition of deferred vs accepted, selective risk
     (accepted vs full accuracy/wF1, accept rate), and whether unhealthy is
     over-deferred.
  3. bothcells hypothesis — confirmed directly from the label files. If bothcells
     is absent from both annotated sets, the hypothesis is UNTESTABLE with this
     data; reported as a limitation, no proxy constructed.

Deferral is defined as prediction-set size != 1 (so empty sets are deferred too);
this differs from run_04's size>1 default, so defer/accept rates here are not
identical to conformal_results_lac.json — but coverage (true label in set) is
independent of that definition and is reproduced exactly (LAC is deterministic).
"""
import sys, csv, json, argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from run_04_conformal import load_all, rank_average_distribution, simple_average_distribution, CLASS_NAMES
from ps3c_robust.selective.conformal import ConformalPredictor

SEED = 42
ALPHAS = [0.05, 0.10, 0.15, 0.20]


def lac_sets(P_cal, y_cal, P_target, alpha):
    """LAC prediction sets: calibrate on (P_cal, y_cal), predict on P_target."""
    cp = ConformalPredictor(alpha=alpha, method="lac", n_classes=3,
                            rng=np.random.default_rng(SEED))
    cp.calibrate(P_cal, y_cal)
    return cp.predict(P_target).prediction_sets           # (N, 3) bool


def coverage(sets, y):
    return float(sets[np.arange(len(y)), y].mean())


def analyse(sets, y, point):
    """Full deferred-set analysis for one (sets, labels) partition."""
    N = len(y)
    size = sets.sum(axis=1)
    accepted = size == 1
    deferred = size != 1
    n_acc, n_def = int(accepted.sum()), int(deferred.sum())

    def comp(mask):
        yy = y[mask]; c = np.bincount(yy, minlength=3); tot = int(c.sum())
        return {CLASS_NAMES[k]: {"n": int(c[k]),
                                 "frac": round(float(c[k] / tot), 4) if tot else None}
                for k in range(3)}

    return {
        "coverage": round(coverage(sets, y), 4),
        "accept_rate": round(n_acc / N, 4),
        "defer_rate": round(n_def / N, 4),
        "set_size_counts": {
            "empty_0": int((size == 0).sum()), "singleton_1": int((size == 1).sum()),
            "doubleton_2": int((size == 2).sum()), "full_3": int((size == 3).sum()),
        },
        "full_accuracy": round(float((point == y).mean()), 4),
        "full_wf1": round(float(f1_score(y, point, average="weighted")), 4),
        "accepted_accuracy": round(float((point[accepted] == y[accepted]).mean()), 4) if n_acc else None,
        "accepted_wf1": round(float(f1_score(y[accepted], point[accepted], average="weighted")), 4) if n_acc else None,
        "deferred_composition": comp(deferred),
        "accepted_composition": comp(accepted),
        "unhealthy_fraction": {
            "overall": round(float((y == 1).mean()), 4),
            "deferred": round(float((y[deferred] == 1).mean()), 4) if n_def else None,
            "accepted": round(float((y[accepted] == 1).mean()), 4) if n_acc else None,
        },
    }


def label_counts(labels_dir, split):
    p = Path(labels_dir) / f"isbi2025-ps3c-{split}-dataset-annotated.csv"
    c = {}
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = row["label"].strip().lower(); c[k] = c.get(k, 0) + 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("STAGE 3 — CONFORMAL SELECTIVE PREDICTION ANALYSIS (LAC, 3-class, canonical)")
    print("=" * 84)

    P_te, y_te = load_all(args.data_dir, args.labels_dir, "test")
    P_ev, y_ev = load_all(args.data_dir, args.labels_dir, "eval")
    ens = {
        "rank_average":   (rank_average_distribution(P_te), rank_average_distribution(P_ev)),
        "simple_average": (simple_average_distribution(P_te), simple_average_distribution(P_ev)),
    }
    print(f"canonical set: test {len(y_te)}  eval {len(y_ev)}")
    for n, (te, ev) in ens.items():
        print(f"  {n:>14} point wF1 — test {f1_score(y_te, te.argmax(1), average='weighted'):.4f}"
              f"  eval {f1_score(y_ev, ev.argmax(1), average='weighted'):.4f}")

    # eval split for within-eval recalibration (Experiment 1)
    idx = np.arange(len(y_ev))
    e_cal, e_hold = train_test_split(idx, test_size=0.5, random_state=SEED, stratify=y_ev)
    print(f"eval split: calibration {len(e_cal)}  hold-out {len(e_hold)}")

    results = {
        "method": "lac", "canonical_n": {"test": int(len(y_te)), "eval": int(len(y_ev))},
        "alphas": ALPHAS, "eval_split": {"calibration": int(len(e_cal)), "holdout": int(len(e_hold))},
        "deferral_definition": "prediction-set size != 1 (empty sets are deferred)",
        "experiment_1_shift_attribution": {}, "experiment_2_deferred_set": {},
        "experiment_3_bothcells": {}, "discrepancy_check_vs_conformal_results_lac": {},
    }

    # ── EXPERIMENT 1 ──────────────────────────────────────────────────────────
    print("\n" + "-" * 84)
    print("EXPERIMENT 1 — shift attribution (coverage on the SAME eval hold-out)")
    print("test-calibrated = cross-split (shifted) | eval-calibrated = within-eval (exchangeable)")
    print("-" * 84)
    print(f"{'ensemble':>14} {'alpha':>6} {'target':>7} {'test-cal(hold)':>15} {'eval-cal(hold)':>15} {'recovered?':>11}")
    for name, (Pte_e, Pev_e) in ens.items():
        block = {"test_calibrated_full_eval": {}, "test_calibrated_eval_holdout": {}, "eval_calibrated_eval_holdout": {}}
        for a in ALPHAS:
            cov_full = coverage(lac_sets(Pte_e, y_te, Pev_e, a), y_ev)
            cov_tc = coverage(lac_sets(Pte_e, y_te, Pev_e[e_hold], a), y_ev[e_hold])
            cov_ec = coverage(lac_sets(Pev_e[e_cal], y_ev[e_cal], Pev_e[e_hold], a), y_ev[e_hold])
            block["test_calibrated_full_eval"][f"{a:.2f}"] = round(cov_full, 4)
            block["test_calibrated_eval_holdout"][f"{a:.2f}"] = round(cov_tc, 4)
            block["eval_calibrated_eval_holdout"][f"{a:.2f}"] = round(cov_ec, 4)
            rec = "YES" if cov_ec >= (1 - a) - 0.01 else "no"
            print(f"{name:>14} {a:>6.2f} {1-a:>7.2f} {cov_tc:>15.4f} {cov_ec:>15.4f} {rec:>11}")
        results["experiment_1_shift_attribution"][name] = block

    # ── EXPERIMENT 2 ──────────────────────────────────────────────────────────
    print("\n" + "-" * 84)
    print("EXPERIMENT 2 — deferred set (test-calibrated on full eval = deployment)")
    print("-" * 84)
    print(f"{'ensemble':>14} {'alpha':>6} {'accept%':>8} {'acc.wF1':>8} {'full.wF1':>8} "
          f"{'unh% overall/defer/accept':>26}")
    for name, (Pte_e, Pev_e) in ens.items():
        block = {}
        point = Pev_e.argmax(1)
        for a in ALPHAS:
            sets = lac_sets(Pte_e, y_te, Pev_e, a)
            m = analyse(sets, y_ev, point)
            block[f"{a:.2f}"] = m
            uf = m["unhealthy_fraction"]
            aw = "n/a" if m["accepted_wf1"] is None else f"{m['accepted_wf1']:.4f}"
            print(f"{name:>14} {a:>6.2f} {100*m['accept_rate']:>7.2f}% {aw:>8} {m['full_wf1']:>8.4f} "
                  f"  {uf['overall']:.3f} / {str(uf['deferred']):>5} / {str(uf['accepted']):>5}")
        results["experiment_2_deferred_set"][name] = block

    # ── EXPERIMENT 3 ──────────────────────────────────────────────────────────
    ct, ce = label_counts(args.labels_dir, "test"), label_counts(args.labels_dir, "eval")
    both = ct.get("bothcells", 0) + ce.get("bothcells", 0)
    results["experiment_3_bothcells"] = {
        "test_label_counts": ct, "eval_label_counts": ce,
        "bothcells_in_test": ct.get("bothcells", 0), "bothcells_in_eval": ce.get("bothcells", 0),
        "testable": both > 0,
        "conclusion": ("bothcells present — hypothesis testable" if both > 0 else
                       "bothcells is ABSENT from both annotated label sets, so the hypothesis "
                       "'bothcells concentrate in the deferred region' is UNTESTABLE with this "
                       "data. Reported as a limitation; no proxy constructed."),
    }
    print("\n" + "-" * 84)
    print("EXPERIMENT 3 — bothcells hypothesis")
    print("-" * 84)
    print(f"  bothcells in test labels: {ct.get('bothcells',0)}   eval labels: {ce.get('bothcells',0)}")
    print(f"  -> {results['experiment_3_bothcells']['conclusion']}")

    # ── discrepancy check vs existing file ────────────────────────────────────
    prev_path = out / "conformal_results_lac.json"
    if prev_path.exists():
        prev = json.loads(prev_path.read_text())
        chk = {}
        for name in ens:
            prev_cov = {a: prev["cross_split_test_to_eval"][name][a]["empirical_coverage"] for a in ["0.05", "0.10", "0.15", "0.20"]}
            new_cov = results["experiment_1_shift_attribution"][name]["test_calibrated_full_eval"]
            max_diff = max(abs(prev_cov[a] - new_cov[a]) for a in prev_cov)
            chk[name] = {"previous": prev_cov, "reproduced": new_cov,
                         "max_abs_diff": round(max_diff, 6),
                         "match": max_diff < 1e-3}
        results["discrepancy_check_vs_conformal_results_lac"] = chk
        print("\ndiscrepancy check vs conformal_results_lac.json (cross-split full-eval coverage):")
        for name, c in chk.items():
            print(f"  {name}: max|Δ|={c['max_abs_diff']}  match={c['match']}")

    (out / "conformal_stage3_analysis.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out/'conformal_stage3_analysis.json'}")
    print("=" * 84)


if __name__ == "__main__":
    main()
