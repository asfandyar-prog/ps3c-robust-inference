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
1. **Method.** APS as implemented in `conformal.py` randomizes the *calibration*
   score but builds prediction sets *non-randomized* — an asymmetry that shrinks
   q̂ and **under-covers** even within-test. LAC has no such asymmetry and meets
   coverage, so it is the default. `--method aps` reproduces the under-coverage.
2. **Representation.** Rank-averaging is invariant to the test→eval shift (its
   per-class marginals are near-identical across splits: `[0.33,0.33,0.35]` vs
   `[0.33,0.33,0.35]`), so it *cannot* exhibit under-coverage. Simple-average
   tracks the shift (`[0.35,0.06,0.60]` → `[0.43,0.11,0.46]`) and does. This
   corroborates why rank-averaging is the robust point-prediction champion.

## Files

Tracked (JSON summaries): `ensemble_baselines.json`, `weight_generator_results.json`,
`conformal_results.json`.
Ignored binaries (regenerate by re-running the scripts): `weight_generator_cluster.pt`,
`cluster_routing_eval.npy`.

Reproduce:
```
python scripts/run_02_baselines.py       --data-dir <team-data> --labels-dir <labels> --out-dir results
python scripts/run_03_weight_generator.py --data-dir <team-data> --labels-dir <labels> --out-dir results --n-clusters 8 --epochs 100
```
(On Windows, prefix with `PYTHONUTF8=1` so console glyphs encode correctly.)
