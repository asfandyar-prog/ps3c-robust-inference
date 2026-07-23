"""
Run (venv Python 3.11):
  python scripts/run_06_bbse_matched.py ^
      --data-dir   E:\ps3c\ps3c-team-data ^
      --labels-dir E:\ps3c\ps3c-labels ^
      --out-dir    E:\ps3c\ps3c-robust-inference\results

Follow-up to the BBSE run (commit 9b689cc). Closes the train/serve input-mismatch
objection: in that run the meta-learners were trained on RAW test probs but
evaluated on BBSE-CORRECTED eval probs, so their collapse (XGB -0.106, LGB -0.100)
was a mismatch artifact, not evidence against correction for learned methods.

Here we RETRAIN the 4 meta-learners on BBSE-corrected TEST probs and evaluate on
BBSE-corrected EVAL probs (matched correction). Per-model shift weights are the
EXACT weights from results/bbse_results.json — loaded, never re-estimated. The
7 base models are NOT retrained. Meta-learner hyperparameters are identical to the
BBSE run (fit_learners), so the only variable across the three columns is which
inputs are corrected:

  (a) raw-trained / raw-eval          [baseline]            <- from bbse_results.json
  (b) raw-trained / corrected-eval    [the mismatch case]   <- from bbse_results.json
  (c) corrected-trained / corrected-eval  [this experiment] <- computed here

Rank averaging (Raw*, 0.8218) is the reference line.
"""
import sys, json, argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                       # import run_05_bbse helpers
sys.path.insert(0, str(HERE.parent / "src"))

from run_05_bbse import load_all_soft, fit_learners, metrics, TEAMS
from ps3c_robust.adapt import correct_probs

LEARNED = ["xgboost", "lightgbm", "catboost", "random_forest"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # exact shift weights + prior results from the BBSE run (not re-estimated)
    bbse = json.loads((out / "bbse_results.json").read_text())
    w = {t: np.array(bbse["bbse_per_model"][t]["weights"], dtype=np.float64) for t in TEAMS}
    prev = bbse["ensemble_raw_vs_corrected"]
    rank_ref = prev["rank_averaging"]["raw_wf1"]

    P_te, y_te = load_all_soft(args.data_dir, args.labels_dir, "test")
    P_ev, y_ev = load_all_soft(args.data_dir, args.labels_dir, "eval")

    # correct BOTH splits with the SAME per-model weights
    Pc_te = np.stack([correct_probs(P_te[:, m], w[TEAMS[m]]) for m in range(len(TEAMS))], axis=1)
    Pc_ev = np.stack([correct_probs(P_ev[:, m], w[TEAMS[m]]) for m in range(len(TEAMS))], axis=1)
    Xc_tr = Pc_te.reshape(len(y_te), -1)
    Xc_ev = Pc_ev.reshape(len(y_ev), -1)

    print("=" * 96)
    print("BBSE MATCHED CORRECTION — meta-learners RETRAINED on corrected test probs")
    print("=" * 96)
    print(f"canonical set: test {len(y_te)}  eval {len(y_ev)}   | shift weights loaded from bbse_results.json (not re-estimated)")
    print(f"reference: rank averaging (Raw*) = {rank_ref:.4f}\n")

    learners = fit_learners(Xc_tr, y_te)             # column (c): corrected-trained

    print(f"{'meta-learner':>14} |  {'(a) raw/raw':^17} |  {'(b) raw/corr':^17} |  {'(c) corr/corr':^17} | {'Δ(c-a)':>8} {'beats':>6}")
    print(f"{'':>14} |  {'wF1':>7} {'unh':>7} |  {'wF1':>7} {'unh':>7} |  {'wF1':>7} {'unh':>7} | {'wF1':>8} {'0.8218':>6}")
    print("-" * 96)

    results = {}
    for name in LEARNED:
        cwf1, crec = metrics(y_ev, learners[name].predict(Xc_ev))
        cwf1 = round(cwf1, 4); crec_u = round(crec["unhealthy"], 4)
        a = prev[name]
        beats = cwf1 > rank_ref
        results[name] = {
            "a_raw_raw": {"wf1": a["raw_wf1"], "unhealthy_recall": a["raw_unhealthy_recall"]},
            "b_raw_corrected": {"wf1": a["corrected_wf1"], "unhealthy_recall": a["corrected_unhealthy_recall"]},
            "c_corrected_corrected": {"wf1": cwf1, "unhealthy_recall": crec_u},
            "delta_c_minus_a": round(cwf1 - a["raw_wf1"], 4),
            "delta_c_minus_b": round(cwf1 - a["corrected_wf1"], 4),
            "beats_rank_avg": bool(beats),
        }
        print(f"{name:>14} |  {a['raw_wf1']:>7.4f} {a['raw_unhealthy_recall']:>7.4f} | "
              f" {a['corrected_wf1']:>7.4f} {a['corrected_unhealthy_recall']:>7.4f} | "
              f" {cwf1:>7.4f} {crec_u:>7.4f} | {cwf1 - a['raw_wf1']:>+8.4f} {('YES' if beats else 'no'):>6}")

    print("-" * 96)
    any_beat = any(results[n]["beats_rank_avg"] for n in LEARNED)
    best = max(LEARNED, key=lambda n: results[n]["c_corrected_corrected"]["wf1"])
    bestv = results[best]["c_corrected_corrected"]["wf1"]
    print(f"\nBest matched-corrected meta-learner: {best} = {bestv:.4f}   "
          f"(rank averaging {rank_ref:.4f})")
    if any_beat:
        print("Result: at least one matched-corrected meta-learner BEATS rank averaging.")
    else:
        print("Result: NO matched-corrected meta-learner reaches rank averaging (0.8218).")
        print("        Matched correction recovers most of the (b) collapse, but rank")
        print("        averaging remains the champion. Negative result, as expected.")

    payload = {
        "reference_rank_averaging_raw": rank_ref,
        "weights_source": "results/bbse_results.json (exact, not re-estimated)",
        "per_learner": results,
        "best_c": {"name": best, "wf1": bestv},
        "any_c_beats_rank_avg": bool(any_beat),
        "protocol_notes": "matched BBSE correction: meta-learners retrained on corrected test probs, "
                          "evaluated on corrected eval probs; hyperparameters identical to the BBSE run; "
                          "7 base models untouched; DPZ normalized.",
    }
    (out / "bbse_matched_results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nSaved -> {out/'bbse_matched_results.json'}")
    print("=" * 96)


if __name__ == "__main__":
    main()
