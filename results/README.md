# Results — canonical sample set & baseline table

## Canonical sample set (cite this everywhere)

For each split, the evaluation set is the **intersection of image IDs scored by
all 7 teams (YMG, JNG, NGU, CHA, GUP, DPZ, WAN), restricted to annotated
ground-truth labels**. This single rule is implemented identically in
`scripts/run_02_baselines.py` and `scripts/run_03_weight_generator.py`
(`load_all`), so every metric below is computed on the same rows.

| Split | Role | N (canonical) |
|-------|------|--------------:|
| test  | train meta-learners | **18,159** |
| eval  | hold-out evaluation | **29,117** |

All 7 teams scored identical image sets, so the 7-way intersection equals each
team's per-team count (verified by `scripts/run_01_verify.py`, which reproduces
paper Table 2 exactly for all 14 team×split cells).

> **Discrepancy resolved.** A prior rank-averaging value of 0.8199 came from the
> *different* loader in `scripts/train_adaptive_ensemble.py`, not this canonical
> intersection. The canonical rank-averaging number is **0.8157** and both
> `run_02` and `run_03` now agree on it.

## Baseline table — eval-set weighted F1 (hold-out, 29,117 samples)

| Method | eval wF1 | vs rank-avg |
|--------|---------:|------------:|
| Simple Average | 0.7585 | −0.0571 |
| Geometric Mean | 0.7178 | −0.0978 |
| Hard Voting | 0.7817 | −0.0339 |
| **Rank Averaging (canonical baseline)** | **0.8157** | +0.0000 |
| Random Forest | 0.6950 | −0.1207 |
| XGBoost | 0.7899 | −0.0258 |
| LightGBM | 0.7983 | −0.0174 |
| CatBoost | 0.7860 | −0.0297 |

**Key finding:** every *learned* meta-learner (RF, XGB, LGB, CatBoost) falls
below the non-learned rank-averaging baseline — the signature of the
preprocessing-induced test→eval distribution shift.

## Weight Generator (Stage 2, cluster variant, 8 clusters)

| Metric | Value |
|--------|------:|
| Best validation wF1 (on test split) | 0.8738 |
| Eval wF1 (hold-out) | **0.7039** |
| Δ vs rank averaging | −0.1117 |
| Unhealthy-class recall (eval) | 0.080 |

The generator learns a strong in-distribution head (val 0.8738) that collapses
out-of-distribution (eval 0.7039), with unhealthy recall falling to ~8%. This is
the central motivating result for the robustness framework.

## Stage 3 — conformal selective prediction (3-class, LAC)

Driver: `scripts/run_04_conformal.py`, using `ConformalPredictor`
(`src/ps3c_robust/selective/conformal.py`) in 3-class mode (bothcells dropped).
Method: split-conformal **LAC** (Sadinle et al., 2019). The ensemble is fed to the
conformal predictor in two forms — `rank_average` (canonical champion) and
`simple_average` (mean of the raw team probabilities). The test split is divided
50/50 into calibration (9,079) / hold-out (9,080), stratified, seed 42.

**Scenario 1 — within-test (exchangeable): the guarantee holds.** Both
representations achieve empirical coverage within ±0.006 of 1−α for
α ∈ {0.05, 0.10, 0.15, 0.20}.

**Scenario 2 — cross-split (calibrate on test → predict on eval):**

| α | target | rank_average cov | simple_average cov |
|--:|-------:|-----------------:|-------------------:|
| 0.05 | 0.95 | 0.998 (+0.048) | **0.936 (−0.013)** |
| 0.10 | 0.90 | 0.978 (+0.077) | **0.844 (−0.056)** |
| 0.15 | 0.85 | 0.940 (+0.090) | **0.767 (−0.083)** |
| 0.20 | 0.80 | 0.892 (+0.092) | **0.700 (−0.101)** |

`simple_average` **under-covers at every α** (the selective-prediction evidence
for the shift); `rank_average` over-covers because it is shift-invariant.

**Two methodological findings (flagged for review):**
1. **Method.** APS originally under-covered even within-test: `conformal.py`
   randomized the *calibration* score but built prediction sets *non-randomized*.
   Fixed `_build_sets` to use the randomized APS inclusion rule (Romano et al.,
   2020), so **both LAC and APS now meet the within-test guarantee** (compare
   `conformal_results_lac.json` and `conformal_results_aps.json`). LAC remains the
   default and gives the cleaner shift demonstration; APS builds larger adaptive
   sets that stay more conservative under the shift.
2. **Representation.** Rank-averaging is invariant to the test→eval shift (its
   per-class marginals are near-identical across splits: `[0.33,0.33,0.35]` vs
   `[0.33,0.33,0.35]`), so it *cannot* exhibit under-coverage. Simple-average
   tracks the shift (`[0.35,0.06,0.60]` → `[0.43,0.11,0.46]`) and does. This
   corroborates why rank-averaging is the robust point-prediction champion.

## BBSE label-shift correction (3-class, normalized DPZ)

Driver: `scripts/run_05_bbse.py`, functions in `src/ps3c_robust/adapt/bbse.py`
(BBSE-hard: joint confusion matrix from argmax test predictions vs true test
labels → least-squares weights → clip ≥0 → target prior; correction
`p(y|x)·w(y)`). Post-hoc on saved probabilities only — no model is retrained.
DPZ (no softmax) is normalized to a pseudo-distribution and used consistently in
the raw and corrected ensembles. Results in `bbse_results.json`.

**Step 2 — the label-shift hypothesis is confirmed, but per-model estimates are
noisy.** True unhealthy prior rises 0.0317 → 0.1896 (5.98×). BBSE's *mean*
estimated eval prior recovers the direction (unhealthy 0.0317 → 0.1383, ~4.4×;
L1 error vs truth 0.1026), but per-model estimates scatter badly because the
unhealthy column of every confusion matrix rests on only 576 test samples with
low model recall:

| model | est. unhealthy ratio q/p_test | (true 5.98×) |
|---|--:|--|
| YMG 7.81 · JNG 10.68 · CHA 7.11 · WAN 3.73 (over/under) · NGU 0.53 · GUP 0.67 · **DPZ 0.00** (weight clipped) | | |

**Step 4 — mixed, and no new champion.** Eval wF1 (Raw* = same pipeline, no
correction; normalized DPZ + no eval early-stopping, so Raw* ≠ published):

| Method | Published | Raw* | Corrected | Δ | unhealthy recall raw→corr |
|---|--:|--:|--:|--:|--|
| Rank Averaging | 0.8157 | **0.8218** | 0.8208 | −0.001 | 0.668 → 0.645 |
| CatBoost | 0.7860 | 0.7901 | 0.7530 | −0.037 | 0.638 → 0.336 |
| Hard Voting | 0.7817 | 0.7859 | **0.8122** | **+0.026** | 0.292 → 0.492 |
| Simple Average | 0.7585 | 0.7713 | **0.8113** | **+0.040** | 0.249 → 0.421 |
| LightGBM | 0.7983 | 0.7826 | 0.6830 | −0.100 | 0.447 → 0.014 |
| XGBoost | 0.7899 | 0.7836 | 0.6771 | −0.106 | 0.439 → 0.003 |
| Geometric Mean | 0.7178 | 0.7296 | 0.7011 | −0.028 | 0.162 → 0.065 |
| Random Forest | 0.6950 | 0.6906 | 0.6738 | −0.017 | 0.095 → 0.000 |

Takeaways:
* **BBSE helps the simple posterior-averaging methods** — simple average +0.040
  and hard voting +0.026, each roughly *doubling* unhealthy recall — which is the
  theoretically valid use (average BBSE-corrected posteriors).
* **BBSE hurts the learned meta-learners** (XGB −0.106, LGB −0.100): feeding
  BBSE-corrected inputs to a model trained on *uncorrected* inputs is not a valid
  label-shift correction; it pushes features out of the training region and
  collapses unhealthy recall.
* **Rank averaging is unmoved** (shift-invariant) and **remains the champion at
  0.8218** — no corrected method beats it. Normalizing DPZ (vs the cascade
  hardening) is what lifts Raw* rank-averaging from 0.8157 to 0.8218.
* Net: BBSE confirms and partially recovers the shift for weak ensembles but does
  **not** produce a new best method. A pooled/ensemble-level BBSE weight (instead
  of noisy per-model weights) is the natural next experiment.

**Matched-correction follow-up** (`scripts/run_06_bbse_matched.py`,
`bbse_matched_results.json`): retraining the 4 meta-learners on BBSE-corrected
*test* probs and evaluating on corrected eval probs (same weights, loaded not
re-estimated; hyperparameters unchanged; base models untouched) confirms the (b)
collapse was a train/serve **input-mismatch artifact**, not a failure of
correction. Eval wF1 across the three matched conditions:

| meta-learner | (a) raw/raw | (b) raw/corr | (c) corr/corr | unhealthy recall a→c |
|---|--:|--:|--:|--|
| XGBoost | 0.7836 | 0.6771 | **0.7905** | 0.439→0.485 |
| LightGBM | 0.7826 | 0.6830 | 0.7821 | 0.447→0.455 |
| CatBoost | 0.7901 | 0.7530 | 0.7828 | 0.638→0.698 |
| Random Forest | 0.6906 | 0.6738 | 0.7013 | 0.095→0.134 |

Matched correction (c) recovers every learner to ~its raw baseline (XGB/RF
marginally above; LGB/CB flat) and restores unhealthy recall — but **none reaches
rank averaging (0.8218)**. Negative result, as expected: correction doesn't break
the learned methods, it just doesn't beat the shift-invariant champion.

## Files

Tracked (JSON summaries): `ensemble_baselines.json`, `weight_generator_results.json`,
`conformal_results_lac.json`, `conformal_results_aps.json`, `bbse_results.json`,
`bbse_matched_results.json`, `baseline_table.csv`.
Ignored binaries (regenerate by re-running the scripts): `weight_generator_cluster.pt`,
`cluster_routing_eval.npy`.

Reproduce:
```
python scripts/run_02_baselines.py       --data-dir <team-data> --labels-dir <labels> --out-dir results
python scripts/run_03_weight_generator.py --data-dir <team-data> --labels-dir <labels> --out-dir results --n-clusters 8 --epochs 100
```
(On Windows, prefix with `PYTHONUTF8=1` so console glyphs encode correctly.)
