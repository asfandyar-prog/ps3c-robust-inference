# PS3C Results — Stage 2 ensembling & label-shift analysis

Every number below is drawn from a saved file in this directory and cited inline.
**Canonical sample set** = per-split intersection of image IDs scored by all 7
teams (YMG, JNG, NGU, CHA, GUP, DPZ, WAN) ∩ annotated ground-truth labels:
**test 18,159 / eval 29,117**. `run_01_verify.py` reproduces paper Table 2 for all
14 team×split cells to the published four decimals.

---

## a. The finding

Preprocessing induces a **label (prior) shift** between the test and eval splits.
The unhealthy-class prior rises **0.0317 → 0.1896 (5.98×)**, confirmed directly
against ground-truth eval labels (`bbse_results.json`: `true_test_prior`,
`true_eval_prior`; `dataset_audit.json`: test unhealthy 576, eval unhealthy 5,521).

Under this shift, **learned ensembling collapses** — every learned meta-learner
falls below the parameter-free baselines, and the Weight Generator's strong
validation score does not transfer (below). **Parameter-free rank averaging is
robust by construction**: it is (near-)invariant to the shift and **remains the
champion at 0.8218 eval wF1** on the normalized-DPZ pipeline (`bbse_results.json`),
equivalently **0.8157** on the published cascade-DPZ pipeline
(`ensemble_baselines.json`).

## b. Baseline ensemble table

Published wF1 from `ensemble_baselines.json`; Raw* wF1 and unhealthy recall from
`bbse_results.json` (`ensemble_raw_vs_corrected.*.raw_wf1` / `raw_unhealthy_recall`).
**Raw\*** = the same 7 models with DPZ normalized to a distribution and
meta-learners trained on raw test only (no eval early-stopping), so it differs
slightly from the published column. The published column has no stored per-class
recall; recall shown is Raw*.

| Method | Published wF1 | Raw* wF1 | Raw* unhealthy recall |
|---|--:|--:|--:|
| **Rank Averaging** | **0.8157** | **0.8218** | 0.668 |
| LightGBM | 0.7983 | 0.7826 | 0.447 |
| XGBoost | 0.7899 | 0.7836 | 0.439 |
| CatBoost | 0.7860 | 0.7901 | 0.638 |
| Hard Voting | 0.7817 | 0.7859 | 0.292 |
| Simple Average | 0.7585 | 0.7713 | 0.249 |
| Geometric Mean | 0.7178 | 0.7296 | 0.162 |
| Random Forest | 0.6950 | 0.6906 | 0.095 |

Every *learned* meta-learner (RF, XGB, LGB, CatBoost) sits below rank averaging —
the signature of the shift.

## c. Weight Generator variants — two DISTINCT models

These are **two different Stage-2 models**, not one:

1. **WeightGeneratorCluster** (`scripts/run_03_weight_generator.py`; cluster-routing
   attention, 8 clusters). Source `weight_generator_results.json`; provenance
   `provenance/weight_generator_rerun.log`:
   - best **validation** wF1 (test 80/20 split): **0.8738**
   - eval wF1: **0.7039** (Δ −0.1117 vs rank averaging)
   - eval **unhealthy recall: 0.0801**
   The high validation wF1 (0.8738) beside 8% unhealthy recall on eval indicates
   **collapse on the shifted class, not genuine performance**: it fits the
   in-distribution (low-unhealthy) validation split, then fails to recognize the
   eval-inflated unhealthy class.

2. **AdaptiveEnsemble** (`src/ps3c_robust/ensemble/adaptive.py`; attention head
   over teams). eval wF1 **0.7479**. ⚠️ Source
   `ps3c-results/adaptive_ensemble_results.txt` (2026-06-18) — computed on a
   **non-canonical** sample set (its rank-averaging reference is 0.8199, not the
   canonical 0.8157) and **not re-run** on the canonical 18,159/29,117 set. Report
   as indicative only; not directly comparable to the tables above.

## d. BBSE label-shift correction

Driver `scripts/run_05_bbse.py`; functions `src/ps3c_robust/adapt/bbse.py`
(BBSE-hard: joint confusion matrix from argmax test predictions vs true test
labels → least-squares weights → clip ≥0 → target prior; correction `p(y|x)·w(y)`).
Post-hoc on saved probabilities only — no retraining. DPZ normalized. Source
`bbse_results.json`.

BBSE's *mean* estimated eval prior recovers the shift direction (unhealthy
0.0317 → 0.1383, ~4.4×; L1 error vs truth 0.1026), but per-model estimates are
noisy (see limitations). Corrected ensemble eval wF1 (Δ vs Raw*):

| Method | Raw* | Corrected | Δ | unhealthy recall raw→corr |
|---|--:|--:|--:|--|
| Rank Averaging | 0.8218 | 0.8208 | −0.001 | 0.668 → 0.645 |
| Simple Average | 0.7713 | **0.8113** | **+0.040** | 0.249 → 0.421 |
| Hard Voting | 0.7859 | **0.8122** | **+0.026** | 0.292 → 0.492 |
| Geometric Mean | 0.7296 | 0.7011 | −0.028 | 0.162 → 0.065 |
| CatBoost | 0.7901 | 0.7530 | −0.037 | 0.638 → 0.336 |
| Random Forest | 0.6906 | 0.6738 | −0.017 | 0.095 → 0.000 |
| LightGBM | 0.7826 | 0.6830 | −0.100 | 0.447 → 0.014 |
| XGBoost | 0.7836 | 0.6771 | −0.106 | 0.439 → 0.003 |

BBSE **helps simple posterior averaging** — simple average **+0.040**, hard voting
**+0.026**, each roughly doubling unhealthy recall (the theoretically valid use:
averaging BBSE-corrected posteriors). It **harms the meta-learners** here because
corrected eval inputs are fed to models trained on *uncorrected* inputs — a
train/serve mismatch, addressed in (e). No corrected method beats rank averaging.

## e. Matched-correction ablation (closes the mismatch objection)

Driver `scripts/run_06_bbse_matched.py`; source `bbse_matched_results.json`.
Meta-learners **retrained on BBSE-corrected test probs** and evaluated on corrected
eval probs, using the **exact** per-model weights from `bbse_results.json` (loaded,
not re-estimated), hyperparameters unchanged, 7 base models untouched. Eval wF1:

| meta-learner | (a) raw/raw | (b) raw/corr | (c) corr/corr | unhealthy recall a→c |
|---|--:|--:|--:|--|
| XGBoost | 0.7836 | 0.6771 | **0.7905** | 0.439 → 0.485 |
| LightGBM | 0.7826 | 0.6830 | 0.7821 | 0.447 → 0.455 |
| CatBoost | 0.7901 | 0.7530 | 0.7828 | 0.638 → 0.698 |
| Random Forest | 0.6906 | 0.6738 | 0.7013 | 0.095 → 0.134 |

The (b) collapse was a **train/serve input-mismatch artifact**: matched training
(c) recovers every learner to ~its raw baseline (XGB/RF marginally above; LGB/CB
flat) and restores unhealthy recall — but **none reaches rank averaging (0.8218)**.
Negative result, as expected: correction neither breaks the learned methods nor
beats the shift-invariant champion.

## f. Known limitations (plainly)

Sources: `bbse_results.json`, `dataset_audit.json`.

- **BBSE unhealthy weight is ill-conditioned.** Only **576** test unhealthy samples
  (`dataset_audit.json`) with low model recall. Per-model unhealthy shift weight
  spans **0.00× (DPZ, clipped to zero) / 0.53× (NGU) to 10.68× (JNG)** against the
  true 5.98× (`bbse_results.json` → `bbse_per_model[*].weights[1]`). Mean-estimate
  L1 error 0.1026. (Note: the low end is 0.00×, not 0.53× — DPZ's weight is clipped
  to zero; 0.53× is NGU, the lowest non-clipped.)
- **DPZ emits independent per-class scores, not a distribution.** Only 15.4% of
  test / 17.9% of eval rows sum to ~1 (`dataset_audit.json`); normalized to a
  pseudo-distribution for consistency across raw and corrected pipelines. Its BBSE
  unhealthy weight is clipped to 0.
- **NGU is near one-hot** — 95.4% of test / 94.8% of eval predictions have
  max-prob > 0.999 (`dataset_audit.json`); posterior correction has minimal effect.
- **bothcells is absent from both annotated label sets** — all evaluation is
  3-class (`dataset_audit.json`: `bothcells_present_in_label_sets = false`; counts
  contain only healthy/unhealthy/rubbish).
- **run_03 provenance.** `run_03_weight_generator.py` was untracked before
  2026-07-02, so its pre-July history is not in git. All surviving copies have
  behavior-identical attention, and the result reproduces bit-identically (eval
  `0.7039267226916234` on 2026-07-02 and again this session; see
  `provenance/weight_generator_rerun.log`). No attention-mechanism fix is recorded
  anywhere for this model; the only "attention" edits in git history are no-op
  whitespace changes to the *separate* `AdaptiveEnsemble`.

## g. Reproduction — script → numbers → file

| Numbers | Script | Results file |
|---|---|---|
| Table 2 reproduction (14 cells) | `run_01_verify.py` | console (git history) |
| Baseline ensemble (published wF1) | `run_02_baselines.py` | `ensemble_baselines.json`, `baseline_table.csv` |
| Weight Generator 0.8738 / 0.7039 / 0.0801 | `run_03_weight_generator.py` | `weight_generator_results.json`, `provenance/weight_generator_rerun.log` |
| Conformal coverage | `run_04_conformal.py` | `conformal_results_lac.json`, `conformal_results_aps.json` |
| BBSE priors/weights + raw-vs-corrected | `run_05_bbse.py` | `bbse_results.json` |
| Matched-correction (a/b/c) | `run_06_bbse_matched.py` | `bbse_matched_results.json` |
| Dataset audit (soft stats, class counts) | audit script | `dataset_audit.json` |
| AdaptiveEnsemble 0.7479 (non-canonical) | `train_adaptive_ensemble.py` | `ps3c-results/adaptive_ensemble_results.txt` |

All runs use venv Python 3.11; on Windows prefix `PYTHONUTF8=1`. Example:
```
python scripts/run_03_weight_generator.py --data-dir <team-data> --labels-dir <labels> --out-dir results --n-clusters 8 --epochs 100
```

---

## Stage 3 — conformal selective prediction (kept for completeness)

Driver `scripts/run_04_conformal.py`, using `ConformalPredictor`
(`src/ps3c_robust/selective/conformal.py`) in 3-class mode. Method LAC (default);
APS also valid after fixing a randomization asymmetry. Test split 50/50 →
calibration/hold-out, seed 42. Within-test coverage holds (±0.006 of 1−α).
Cross-split (calibrate test → predict eval), simple-average ensemble
under-covers — the selective-prediction signature of the shift
(`conformal_results_lac.json`):

| α | target | rank_average cov | simple_average cov |
|--:|--:|--:|--:|
| 0.05 | 0.95 | 0.998 | 0.9365 |
| 0.10 | 0.90 | 0.978 | 0.8436 |
| 0.15 | 0.85 | 0.940 | 0.7668 |
| 0.20 | 0.80 | 0.892 | 0.6995 |

rank_average over-covers (shift-invariant); simple_average under-covers at every α.

## Files

Tracked JSON/text: `ensemble_baselines.json`, `weight_generator_results.json`,
`bbse_results.json`, `bbse_matched_results.json`, `conformal_results_lac.json`,
`conformal_results_aps.json`, `dataset_audit.json`, `baseline_table.csv`,
`provenance/weight_generator_rerun.log`.
Ignored binaries (regenerate by re-running): `weight_generator_cluster.pt`,
`cluster_routing_eval.npy`.
