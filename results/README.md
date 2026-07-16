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

## Files

Tracked (JSON summaries): `ensemble_baselines.json`, `weight_generator_results.json`.
Ignored binaries (regenerate by re-running the scripts): `weight_generator_cluster.pt`,
`cluster_routing_eval.npy`.

Reproduce:
```
python scripts/run_02_baselines.py       --data-dir <team-data> --labels-dir <labels> --out-dir results
python scripts/run_03_weight_generator.py --data-dir <team-data> --labels-dir <labels> --out-dir results --n-clusters 8 --epochs 100
```
(On Windows, prefix with `PYTHONUTF8=1` so console glyphs encode correctly.)
