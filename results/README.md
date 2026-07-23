# PS3C Results — two-stage framework (Stages 2 & 3)

This is a **two-stage framework**: shift-robust ensembling (Stage 2) and conformal
selective prediction (Stage 3). The stages keep the numbers **2** and **3** from
the project's original three-stage design so the labels stay aligned with the
`run_0N` scripts, these results files, and the code (e.g. `conformal.py` refers to
"Stage 3"). Architecture-aware TTA (the former "Stage 1") is **out of scope** —
implemented (`src/ps3c_robust/tta/`) but not evaluated (see the repository README).

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

2. **AdaptiveEnsemble** (`scripts/train_adaptive_ensemble.py`,
   `src/ps3c_robust/ensemble/adaptive.py`; attention head over teams). Re-run on
   the **canonical loader** — identical to run_03 (7-way intersection, DPZ
   hardened via the cascade rule); model architecture and hyperparameters
   unchanged, only the data-loading path. Source `adaptive_ensemble_canonical.json`:
   - eval wF1 **0.7364**, eval **unhealthy recall 0.1541**
   - **loader sanity check:** rank averaging comes out at the canonical **0.8157**,
     and simple-average (0.7585) / hard-voting (0.7817) reproduce the baseline
     table exactly — confirming the loader matches run_03.
   > Supersedes the earlier **0.7479** (`ps3c-results/adaptive_ensemble_results.txt`,
   > 2026-06-18), which used a different loader (soft DPZ, rank-avg reference
   > 0.8199, not re-run on the canonical set).

**Stage-2 models on equal footing (canonical loader, eval split):**

| Model | eval wF1 | unhealthy recall |
|---|--:|--:|
| Rank Averaging (non-learned champion) | **0.8157** | — |
| AdaptiveEnsemble | 0.7364 | 0.1541 |
| WeightGeneratorCluster | 0.7039 | 0.0801 |

Both learned Stage-2 models collapse below parameter-free rank averaging, and
neither recovers the unhealthy class (recall 0.15 / 0.08 against the eval prior
0.19). (The comparison uses the canonical rank-averaging value **0.8157**; the
**0.8218** cited under BBSE is the *normalized-DPZ* variant from `run_05`, a
different DPZ representation. Rank averaging's per-class recall is not persisted
for the canonical loader; on the normalized-DPZ pipeline it is 0.668,
`bbse_results.json`.)

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
| Conformal Stage-3 analysis (Exp 1–3) | `run_07_conformal_analysis.py` | `conformal_stage3_analysis.json` |
| BBSE priors/weights + raw-vs-corrected | `run_05_bbse.py` | `bbse_results.json` |
| Matched-correction (a/b/c) | `run_06_bbse_matched.py` | `bbse_matched_results.json` |
| Dataset audit (soft stats, class counts) | audit script | `dataset_audit.json` |
| AdaptiveEnsemble 0.7364 (canonical) | `train_adaptive_ensemble.py` | `adaptive_ensemble_canonical.json` |

All runs use venv Python 3.11; on Windows prefix `PYTHONUTF8=1`. Example:
```
python scripts/run_03_weight_generator.py --data-dir <team-data> --labels-dir <labels> --out-dir results --n-clusters 8 --epochs 100
```

---

## Stage 3 — conformal selective prediction

Predictor `src/ps3c_robust/selective/conformal.py`; coverage tables from
`scripts/run_04_conformal.py`, the analysis below from
`scripts/run_07_conformal_analysis.py`. **LAC** scoring, 3-class, canonical loader,
seed 42. Analysed on two ensembles: **rank_average** (champion, eval point wF1
0.8157) and **simple_average** (eval 0.7585). Sources: `conformal_results_lac.json`
(coverage) and `conformal_stage3_analysis.json` (Experiments 1–3). Deferral is
defined as **prediction-set size ≠ 1** (empty sets are deferred too).

> The cross-split coverage numbers reproduce `conformal_results_lac.json` **exactly**
> (discrepancy check `max |Δ| = 0.0` for both ensembles); the new file *adds*
> analysis and does not overwrite the old one.

### Experiment 1 — the coverage failure is shift, not a bug

Coverage on the **same** eval hold-out (14,559 samples), calibrated two ways:
cross-split (calibrate on full test) vs within-eval (calibrate on the other 14,558
eval samples).

| α (target) | rank_avg test-cal | rank_avg **eval-cal** | simple_avg test-cal | simple_avg **eval-cal** |
|--:|--:|--:|--:|--:|
| 0.05 (0.95) | 0.9980 | **0.9508** | 0.9389 | **0.9554** |
| 0.10 (0.90) | 0.9771 | **0.9016** | 0.8436 | **0.9003** |
| 0.15 (0.85) | 0.9412 | **0.8520** | 0.7656 | **0.8483** |
| 0.20 (0.80) | 0.8928 | **0.7994** | 0.6972 | **0.7991** |

**Recalibrating within eval restores nominal coverage for both ensembles at every
α** → the cross-split behaviour is attributable to the test→eval shift breaking
exchangeability, not to an implementation bug. Two distinct failure modes under the
shift: the shift-invariant `rank_average` **over-covers** (conservative — clinically
safe), while `simple_average` **under-covers** (unsafe — real coverage 0.70 at a 0.80
target). Shift-invariant point predictions imply *conservative* coverage, not nominal.

### Experiment 2 — what is deferred (test-calibrated on full eval = deployment)

Selective risk (accepted = singletons; full = all 29,117):

| ensemble | α | accept % | accepted wF1 | full wF1 |
|---|--:|--:|--:|--:|
| rank_average | 0.05 | 5.7 | 0.9976 | 0.8157 |
| rank_average | 0.10 | 25.7 | 0.9787 | 0.8157 |
| rank_average | 0.15 | 52.8 | 0.9320 | 0.8157 |
| rank_average | 0.20 | 76.4 | 0.8764 | 0.8157 |
| simple_average | 0.05 | 65.0 | 0.9076 | 0.7585 |
| simple_average | 0.10 | 89.5 | 0.7953 | 0.7585 |
| simple_average | 0.15 | 93.6 | 0.7829 | 0.7585 |
| simple_average | 0.20 | 79.8 | 0.8461 | 0.7585 |

Abstaining on non-singletons buys accuracy on the accepted set (the risk–coverage
trade-off). Set-size distribution (N=29,117) shows where deferrals come from: at low
α mostly multi-label sets, at high α **empty sets** appear as the LAC threshold 1−q̂
rises — e.g. `simple_average` at α=0.20 has **5,880 empty / 23,237 singleton / 0
larger**, which is why its accept rate is non-monotonic (93.6% at α=0.15 → 79.8% at
α=0.20). `rank_average` at α=0.05 is the opposite extreme: 1,652 singleton / 17,504
doubleton / 9,961 full, accept 5.7%.

**Clinically relevant finding — unhealthy is over-deferred.** Unhealthy fraction
(eval prior **0.190**) by partition:

| ensemble | α | overall | deferred | accepted |
|---|--:|--:|--:|--:|
| rank_average | 0.05 | 0.190 | 0.199 | 0.037 |
| rank_average | 0.20 | 0.190 | 0.368 | 0.135 |
| simple_average | 0.05 | 0.190 | 0.418 | 0.066 |
| simple_average | 0.15 | 0.190 | 0.566 | 0.164 |

At every α and for both ensembles the deferred set is **enriched** in unhealthy and
the accepted set is **depleted** of it (up to 56.6% unhealthy deferred vs 3.7–16%
accepted). The system abstains disproportionately on the clinically most important
class. This cuts both ways — deferring uncertain unhealthy cases to a cytologist is
arguably the *safe* behaviour, but the automated accept set delivers least value
exactly where it matters most, and this must be stated plainly rather than presented
as unqualified selective-prediction success.

### Experiment 3 — bothcells hypothesis is untestable

The stated hypothesis was that bothcells samples concentrate in the deferred region.
Confirmed directly from the label files (`experiment_3_bothcells`): **bothcells
appears 0 times in both annotated label sets** (test: 5,826 healthy / 576 unhealthy /
11,757 rubbish; eval: 11,609 / 5,521 / 11,987). All evaluation is 3-class, so the
hypothesis is **untestable with this data** — reported as a limitation; no proxy was
constructed.

## Files

Tracked JSON/text: `ensemble_baselines.json`, `weight_generator_results.json`,
`adaptive_ensemble_canonical.json`, `adaptive_ensemble_results.txt`,
`bbse_results.json`, `bbse_matched_results.json`, `conformal_results_lac.json`,
`conformal_results_aps.json`, `conformal_stage3_analysis.json`, `dataset_audit.json`,
`baseline_table.csv`, `provenance/weight_generator_rerun.log`.
Ignored binaries (regenerate by re-running): `weight_generator_cluster.pt`,
`cluster_routing_eval.npy`.
