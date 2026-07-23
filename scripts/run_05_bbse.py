"""
Run from anywhere (venv Python 3.11):
  python scripts/run_05_bbse.py ^
      --data-dir   E:\ps3c\ps3c-team-data ^
      --labels-dir E:\ps3c\ps3c-labels ^
      --out-dir    E:\ps3c\ps3c-robust-inference\results

Stage: BBSE (Black Box Shift Estimation) label-shift correction on the 7 PS3C
base models, applied post-hoc to saved probability outputs (no retraining).

Protocol (see results/README.md "BBSE" section):
  * 3 classes (healthy, unhealthy, rubbish); canonical set = 7-way intersection.
  * DPZ has no softmax output (independent per-class scores) -> normalized to a
    pseudo-distribution and used consistently in BOTH the raw and corrected
    ensembles, so the only raw-vs-corrected difference is the BBSE reweighting.
  * BBSE-hard per model: argmax test predictions vs true test labels -> joint
    confusion matrix -> least-squares weights -> clip >=0 -> target prior.
  * Correction applied to EVAL probs per model, before ensembling.
  * Meta-learners (RF/XGB/LGB/CB) trained on RAW test probs only (no eval_set /
    no early stopping -> no eval-label leakage), then applied to raw & corrected
    eval. "Raw*" therefore uses normalized DPZ and no early stopping, so it can
    differ from the published baseline (cascade DPZ + early stopping), shown as a
    reference row.
  * We DO have eval labels, so BBSE's estimated prior is validated against truth.
"""
import sys, csv, json, argparse
from pathlib import Path
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.metrics import f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

from ps3c_robust.adapt import estimate_target_prior, correct_probs

CANON = {"healthy": 0, "unhealthy": 1, "rubbish": 2}
CLASS_NAMES = ["healthy", "unhealthy", "rubbish"]
TEAMS = ["YMG", "JNG", "NGU", "CHA", "GUP", "DPZ", "WAN"]
ALI = {
    "healthy": "healthy", "healthy_prob": "healthy", "prob_healthy": "healthy",
    "unhealthy": "unhealthy", "unhealthy_prob": "unhealthy", "prob_unhealthy": "unhealthy",
    "rubbish": "rubbish", "rubbish_prob": "rubbish", "prob_rubbish": "rubbish",
}
SEED = 42


def norm(n):
    n = n.strip(); return n[:-4] if n.lower().endswith(".png") else n


def file_map(D):
    D = Path(D)
    return {
        "YMG": {"test": D/"wrapup_UT/wrapup_UT/test_phase_prob.csv", "eval": D/"wrapup_UT/wrapup_UT/final_eval_phase_revised_prob.csv"},
        "JNG": {"test": D/"jianght_challenge_materials/Evaluation-set.csv", "eval": D/"jianght_challenge_materials/Test-set.csv"},
        "NGU": {"test": D/"NGU/predictions_isbi2025-ps3c-test-dataset.csv", "eval": D/"NGU/predictions_isbi2025-ps3c-eval-dataset.csv"},
        "CHA": {"test": D/"ChatoLina_challenge_materials/isbi2025-ps3c-test-dataset pro Ens.csv", "eval": D/"ChatoLina_challenge_materials/isbi2025-ps3c-eval-dataset pro Ens.csv"},
        "GUP": {"test": D/"Chinmay materials/ISBI PS3C IIT KGP/Test_Set_ProbabilityScores.csv", "eval": D/"Chinmay materials/ISBI PS3C IIT KGP/Eval_Set_ProbabilityScores.csv"},
        "DPZ": {"test": D/"Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_test.csv", "eval": D/"Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_eval.csv"},
        "WAN": {"test": D/"WAN/validation_predictions2.csv", "eval": D/"WAN/val_predictions_resnet50.csv"},
    }


def load_labels(labels_dir, split):
    p = Path(labels_dir)/f"isbi2025-ps3c-{split}-dataset-annotated.csv"; gt = {}
    with open(p, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            lab = r["label"].strip().lower()
            if lab in CANON: gt[norm(r["image_name"])] = CANON[lab]
    return gt


def load_team_soft(path):
    """Unified SOFT loader for all 7 teams (incl. DPZ): read the 3 class columns,
    drop bothcells, renormalize each row to a distribution. For DPZ this converts
    its independent per-class scores into a normalized pseudo-distribution."""
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f); header = [h.strip().lower() for h in next(rd)]
        rows = [r for r in rd if r and r[0].strip()]
    idx = {i: ALI[c] for i, c in enumerate(header) if c in ALI}
    d = {}
    for r in rows:
        v = np.zeros(3, dtype=np.float64)
        for i, cls in idx.items():
            try: v[CANON[cls]] = float(r[i])
            except (ValueError, IndexError): pass
        s = v.sum(); d[norm(r[0])] = v/s if s > 0 else v
    return d


def load_all_soft(data_dir, labels_dir, split):
    gt = load_labels(labels_dir, split); fm = file_map(data_dir)
    pt = {t: load_team_soft(fm[t][split]) for t in TEAMS}
    common = set(gt)
    for t in TEAMS: common &= set(pt[t])
    names = sorted(common)
    probs = np.stack([np.array([pt[t][n] for n in names], dtype=np.float64) for t in TEAMS], axis=1)
    labels = np.array([gt[n] for n in names], dtype=np.int64)
    return probs, labels          # probs: (N, 7, 3)


# ── ensemble methods (operate on (N,7,3) probability tensor) ──────────────────
def m_simple(P):  return P.mean(1).argmax(1)
def m_geom(P):    return np.exp(np.log(np.clip(P, 1e-12, None)).mean(1)).argmax(1)
def m_hard(P):
    preds = P.argmax(2)                                    # (N,7)
    out = np.zeros((len(P), 3))
    for t in range(preds.shape[1]):
        out[np.arange(len(P)), preds[:, t]] += 1
    return out.argmax(1)
def m_rank(P):
    N, T, C = P.shape; ranks = np.zeros_like(P)
    for t in range(T):
        for c in range(C):
            order = np.argsort(P[:, t, c]); ranks[order, t, c] = np.arange(N)
    return ranks.mean(1).argmax(1)


def cls_weights(y):
    counts = Counter(y); total = len(y)
    return np.array([total/(3*counts[yi]) for yi in y])


def fit_learners(X_tr, y_tr):
    """Train the four meta-learners on raw test features only (no eval_set)."""
    learners = {}
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(X_tr, y_tr); learners["random_forest"] = rf
    xg = xgb.XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, eval_metric="mlogloss", random_state=SEED,
                           n_jobs=-1, verbosity=0)
    xg.fit(X_tr, y_tr, sample_weight=cls_weights(y_tr)); learners["xgboost"] = xg
    lg = lgb.LGBMClassifier(n_estimators=500, num_leaves=63, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, class_weight="balanced", random_state=SEED,
                            n_jobs=-1, verbose=-1)
    lg.fit(X_tr, y_tr, sample_weight=cls_weights(y_tr)); learners["lightgbm"] = lg
    cb = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, loss_function="MultiClass",
                            class_weights=[1.0, 5.0, 1.0], random_seed=SEED, verbose=0)
    cb.fit(X_tr, y_tr); learners["catboost"] = cb
    return learners


def metrics(y_true, y_pred):
    wf1 = float(f1_score(y_true, y_pred, average="weighted"))
    rec = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2])
    return wf1, {CLASS_NAMES[k]: float(rec[k]) for k in range(3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("=" * 82)
    print("BBSE LABEL-SHIFT CORRECTION — 7 PS3C base models (3-class, normalized DPZ)")
    print("=" * 82)

    P_te, y_te = load_all_soft(args.data_dir, args.labels_dir, "test")
    P_ev, y_ev = load_all_soft(args.data_dir, args.labels_dir, "eval")
    print(f"\ncanonical set: test {len(y_te)}   eval {len(y_ev)}")

    p_test = np.bincount(y_te, minlength=3) / len(y_te)
    p_eval_true = np.bincount(y_ev, minlength=3) / len(y_ev)     # validation target

    # ── STEP 2: per-model BBSE prior estimation + diagnostics ────────────────
    print("\n" + "-" * 82)
    print("STEP 2 — BBSE estimated EVAL prior per model vs TRUE eval prior")
    print("-" * 82)
    print(f"  true test prior : healthy {p_test[0]:.4f}  unhealthy {p_test[1]:.4f}  rubbish {p_test[2]:.4f}")
    print(f"  true eval prior : healthy {p_eval_true[0]:.4f}  unhealthy {p_eval_true[1]:.4f}  rubbish {p_eval_true[2]:.4f}")
    print(f"  {'model':>5} {'q_health':>9} {'q_unhlth':>9} {'q_rubb':>8} {'w_unhlth':>9} {'condC':>8}  unhlth ratio q/p_test")
    weights = {}
    est_priors = {}
    for m, t in enumerate(TEAMS):
        yhat_te = P_te[:, m].argmax(1)
        yhat_ev = P_ev[:, m].argmax(1)
        q, w, C, nu, p_src = estimate_target_prior(y_te, yhat_te, yhat_ev, 3)
        weights[t] = w; est_priors[t] = q
        condC = np.linalg.cond(C)
        print(f"  {t:>5} {q[0]:>9.4f} {q[1]:>9.4f} {q[2]:>8.4f} {w[1]:>9.3f} {condC:>8.1f}   {q[1]/p_test[1]:>6.2f}x  (true {p_eval_true[1]/p_test[1]:.2f}x)")

    q_avg = np.mean([est_priors[t] for t in TEAMS], axis=0)
    print(f"\n  mean estimated eval prior : healthy {q_avg[0]:.4f}  unhealthy {q_avg[1]:.4f}  rubbish {q_avg[2]:.4f}")
    print(f"  TRUE eval prior           : healthy {p_eval_true[0]:.4f}  unhealthy {p_eval_true[1]:.4f}  rubbish {p_eval_true[2]:.4f}")
    l1 = float(np.abs(q_avg - p_eval_true).sum())
    print(f"  L1 error (mean est vs true): {l1:.4f}   -> label-shift hypothesis {'CONFIRMED' if q_avg[1] > 2*p_test[1] else 'NOT supported'} (unhealthy up)")

    # ── STEP 3: correct eval probs per model ─────────────────────────────────
    P_ev_corr = np.stack([correct_probs(P_ev[:, m], weights[TEAMS[m]]) for m in range(len(TEAMS))], axis=1)

    # ── STEP 4: ensemble comparison, raw vs corrected ────────────────────────
    print("\n" + "-" * 82)
    print("STEP 4 — ensemble eval wF1: raw vs BBSE-corrected (+ unhealthy recall)")
    print("-" * 82)

    # load published baseline for reference
    ref = {}
    ref_path = out / "ensemble_baselines.json"
    if ref_path.exists():
        ref = json.loads(ref_path.read_text())

    simple_methods = {"simple_average": m_simple, "hard_voting": m_hard,
                      "rank_averaging": m_rank, "geometric_mean": m_geom}

    X_tr = P_te.reshape(len(y_te), -1)
    X_ev_raw = P_ev.reshape(len(y_ev), -1)
    X_ev_corr = P_ev_corr.reshape(len(y_ev), -1)
    learners = fit_learners(X_tr, y_te)

    order = ["rank_averaging", "lightgbm", "xgboost", "catboost", "hard_voting",
             "simple_average", "geometric_mean", "random_forest"]
    results = {}
    for name in order:
        if name in simple_methods:
            f = simple_methods[name]
            raw_wf1, raw_rec = metrics(y_ev, f(P_ev))
            cor_wf1, cor_rec = metrics(y_ev, f(P_ev_corr))
        else:
            mdl = learners[name]
            raw_wf1, raw_rec = metrics(y_ev, mdl.predict(X_ev_raw))
            cor_wf1, cor_rec = metrics(y_ev, mdl.predict(X_ev_corr))
        results[name] = {"raw_wf1": round(raw_wf1, 4), "corrected_wf1": round(cor_wf1, 4),
                         "delta": round(cor_wf1 - raw_wf1, 4),
                         "raw_unhealthy_recall": round(raw_rec["unhealthy"], 4),
                         "corrected_unhealthy_recall": round(cor_rec["unhealthy"], 4),
                         "published_baseline": ref.get(name)}

    print(f"{'method':>16} {'published':>10} {'Raw*':>8} {'Corrected':>10} {'Δ(C-Raw*)':>10} "
          f"{'unhlthRec raw->corr':>22}")
    for name in order:
        r = results[name]
        pub = "   —  " if r["published_baseline"] is None else f"{r['published_baseline']:.4f}"
        flag = "  ▲" if r["delta"] > 0.001 else ("  ▼" if r["delta"] < -0.001 else "   ·")
        print(f"{name:>16} {pub:>10} {r['raw_wf1']:>8.4f} {r['corrected_wf1']:>10.4f} "
              f"{r['delta']:>+10.4f}{flag} {r['raw_unhealthy_recall']:>10.4f} -> {r['corrected_unhealthy_recall']:<.4f}")

    # ── STEP 5: persist ──────────────────────────────────────────────────────
    payload = {
        "canonical_n": {"test": int(len(y_te)), "eval": int(len(y_ev))},
        "true_test_prior": {CLASS_NAMES[k]: float(p_test[k]) for k in range(3)},
        "true_eval_prior": {CLASS_NAMES[k]: float(p_eval_true[k]) for k in range(3)},
        "bbse_per_model": {t: {"weights": [float(x) for x in weights[t]],
                               "estimated_eval_prior": [float(x) for x in est_priors[t]]} for t in TEAMS},
        "bbse_mean_estimated_eval_prior": {CLASS_NAMES[k]: float(q_avg[k]) for k in range(3)},
        "bbse_prior_l1_error_vs_true": round(l1, 4),
        "ensemble_raw_vs_corrected": results,
        "protocol_notes": "normalized DPZ; meta-learners trained on raw test only (no eval_set); "
                          "Raw* differs from published baseline (cascade DPZ + early stopping).",
    }
    (out / "bbse_results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out/'bbse_results.json'}")
    print("=" * 82)


if __name__ == "__main__":
    main()
