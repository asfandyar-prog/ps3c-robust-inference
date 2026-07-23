<div align="center">

# PS3C Robust Inference

**Architecture-aware test-time adaptation, sample-adaptive ensembling, and conformal selective prediction for cervical cell classification.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#status)
[![uv](https://img.shields.io/badge/managed%20with-uv-orange)](https://docs.astral.sh/uv/)

*A research project extending the PS3C: Pap Smear Cell Classification Challenge (Kupas, Harangi et al.).*

</div>

---

## Status

Active research project under Prof. Balazs Harangi (University of Debrecen). The
official ground-truth labels are now in hand; **all seven** teams' probability
outputs are ingested and aligned, paper Table 2 is reproduced exactly, and
**Stage 2 ensembling plus a label-shift (BBSE) analysis are complete**. Stage 1
(TTA) and Stage 3 (conformal) frameworks are implemented; Stage 1 on real data
remains blocked on the APACC dataset + HPC.

| Component | Status | Notes |
|---|---|---|
| Repository scaffold | Complete | uv + Python 3.11 |
| Official ground-truth labels | **Obtained** | annotated test + eval sets |
| Seven-team probability ingest | **Complete** | all 7 teams aligned by image name, both splits |
| Baseline reproduction (paper Table 2) | **Reproduced** | 14/14 cells to 4 decimals (`run_01_verify.py`) |
| Stage 2 — ensemble baselines + Weight Generator | **Complete** | see Results |
| Label-shift (BBSE) analysis | **Complete** | `run_05` / `run_06`, see Results |
| Stage 3 (conformal) | Implemented + run | `run_04_conformal.py` |
| Stage 1 (TTA) | Implemented | on real data blocked on APACC + HPC |
| Journal / workshop paper | In progress | Target: MIDL / ISBI 2027 |

---

## Results (Stage 2 & label-shift analysis)

Full detail with every number cited to a file: **[`results/README.md`](results/README.md)**.
Canonical sample set = per-split intersection of image IDs scored by all 7 teams ∩
annotated labels: **test 18,159 / eval 29,117**.

**The finding.** Preprocessing induces a **label (prior) shift** between test and
eval: the unhealthy prior rises **0.0317 → 0.1896 (5.98×)**, confirmed against
ground-truth eval labels. Learned ensembling collapses under it; parameter-free
**rank averaging is shift-invariant by construction and remains the champion at
0.8218 eval wF1** (0.8157 on the published cascade-DPZ pipeline).

**Baseline ensemble** (eval wF1; published from `ensemble_baselines.json`, Raw* +
unhealthy recall from `bbse_results.json`):

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

**Weight Generator — two distinct models.** *WeightGeneratorCluster*
(`run_03`, `weight_generator_results.json`): validation wF1 **0.8738** but eval
wF1 **0.7039** with **8.0% unhealthy recall** — the high val score is collapse on
the shifted class, not genuine performance. *AdaptiveEnsemble*
(`train_adaptive_ensemble.py`): eval wF1 **0.7364**, unhealthy recall **0.1541**
on the canonical loader (supersedes the earlier non-canonical 0.7479). Both
learned Stage-2 models sit below rank averaging (0.8157 canonical).

**BBSE label-shift correction** (`run_05`, `bbse_results.json`). Applied post-hoc,
no retraining. Helps simple posterior averaging (simple average **+0.040**, hard
voting **+0.026**, each ~doubling unhealthy recall) and harms the mismatched
meta-learners. The matched-correction ablation (`run_06`, `bbse_matched_results.json`)
shows that collapse was a train/serve input mismatch — retraining on corrected
inputs recovers the learners (best XGBoost 0.7905) — but **no corrected method
beats rank averaging (0.8218)**. Known limitations (BBSE unhealthy weight
ill-conditioned; DPZ non-distributional; NGU near one-hot; 3-class only) are
enumerated in [`results/README.md`](results/README.md).

---

## The problem this work addresses

The PS3C challenge benchmarked seven deep-learning teams on cervical cell classification using the [APACC dataset](https://osf.io/fp2xe/) (103,675 images across 107 patients). The proceedings paper revealed two deployment-relevant issues that the original work documented but did not address.

### Issue 1 — Preprocessing-induced distribution shift

The hidden evaluation set used a slightly different preprocessing pipeline than the test set: cell crops were normalized to 224x224 with white-pixel padding. Every team's macro-F1 dropped on the evaluation set, and the size of the drop depended on whether each team's training already matched the eval-side preprocessing.

The figures reported in the original paper (all seven teams):

| Team / system | Architecture family | Test F1 | Eval F1 | Drop |
|---|---|---|---|---|
| DPZ | Hybrid (CNN+ViT) | 0.8622 | 0.7092 | 0.153 |
| GUP | CNN ensemble | 0.8604 | 0.7211 | 0.139 |
| YMG | MaxViT (padded already) | 0.8680 | 0.7858 | 0.082 |
| JNG | Foundation models + LoRA | 0.8702 | 0.8176 | 0.053 |
| Best ensemble (Gradient Boost stacking) | Meta-learner | 0.9517 | 0.9245 | 0.027 |

The strongest individual models and the meta-learner ensemble lose only a few points, but every system loses something, and no team's reported method actively addresses the shift. Closing this residual gap without retraining and without target-domain labels is the goal of Stage 1.

### Issue 2 — Calibration is unreported

The PS3C paper reports accuracy and macro-F1 but not calibration. This matters clinically: a 92% F1 ensemble that is overconfident on its errors is harder to deploy safely than one whose probabilities reflect its actual reliability.

The connection to the developer's BSc thesis is direct: that work measured a large rise in Expected Calibration Error for supervised ViTs under medical domain shift (0.014 to 0.889 on PathMNIST to DermaMNIST), and showed that LayerNorm-only entropy minimization recovers calibration without target-domain labels. The hypothesis here is that the same mechanism, applied across a heterogeneous ensemble, recovers calibration on the PS3C evaluation set, with Stage 3 turning the calibrated probabilities into formal coverage guarantees.

---

## The three-stage framework

```
            APACC cell image
                   |
                   v
   STAGE 1  Architecture-Aware Test-Time Adaptation
   --------------------------------------------------
     ViT models        CNN models        Hybrid models
     (JNG, YMG)        (GUP, WAN)         (DPZ, CHA)
     LayerNorm TTA     TENT (BatchNorm)   both surfaces
                   |
                   v   adapted probability vectors
   STAGE 2  Sample-Adaptive Ensemble
   --------------------------------------------------
     attention head -> per-sample weights over teams
     (replaces the static gradient-boost weights)
                   |
                   v   fused probabilities
   STAGE 3  Conformal Selective Prediction
   --------------------------------------------------
     split-conformal -> 95% coverage guarantee
     ambiguous cases (incl. bothcells) -> defer
                   |
                   v
            Predict OR Defer
```

### Stage 1 — Architecture-Aware Test-Time Adaptation

Different model families need different adaptation surfaces. TENT (Wang et al., ICLR 2021) targets BatchNorm and does not apply to Vision Transformers. LayerNorm-based TTA does not apply to pure CNNs. PS3C contains both, plus hybrids, so adaptation is dispatched per architecture.

| Family | Mechanism | Applies to |
|---|---|---|
| ViT | Entropy minimization on LayerNorm scale and shift | JNG, YMG |
| CNN | TENT (BatchNorm affine + recomputed statistics) | GUP, WAN |
| Hybrid | Both surfaces together | DPZ, CHA |

The LayerNorm TTA implementation is ported from the developer's BSc thesis. What is new here is the dispatch pattern: applying the correct adaptation surface to each model in a heterogeneous ensemble, and ablating which surface matters more in the hybrid models.

### Stage 2 — Sample-Adaptive Ensemble

A lightweight attention head produces per-sample weights over the team outputs, replacing the original paper's static gradient-boost weights. Per team it sees the softmax probabilities plus simple confidence statistics (max probability, entropy, top-1 minus top-2 margin), and outputs a fused 3-class probability vector together with per-sample weights showing which team was trusted for each prediction.

The hypothesis: different cell types have different best models, and a static ensemble cannot exploit that. Whether it holds is one of the questions the experiments will answer.

### Stage 3 — Conformal Selective Prediction

Split-conformal prediction (APS / LAC scoring) with a target marginal coverage of 95%. The same entropy threshold that gates Stage 1 adaptation also drives the Stage 3 deferral decision, giving the pipeline a single interpretable confidence axis. The system defers when the conformal prediction set has more than one label, or when the ambiguous "bothcells" class is in the set above a configurable threshold. The expectation, testable once the labels are confirmed, is that bothcells samples concentrate in the deferred region.

---

## What is and is not novel

Stated plainly, because it shapes how the work should be reviewed.

**Reused from prior work:**
- Entropy-minimization TTA loss (TENT, Wang et al., 2021)
- LayerNorm parameter targeting (TTT++, Liu et al., 2021)
- APS / LAC conformal scoring (Romano et al., 2020; Sadinle et al., 2019)

**Contributed here:**
- A dispatch pattern that applies the right TTA surface per architecture across a heterogeneous ensemble, with an ablation of LayerNorm vs BatchNorm in the hybrid models.
- ECE as a first-class metric on the PS3C benchmark, which the original challenge does not report.
- A single entropy threshold that gates both adaptation and deferral.
- A bothcells-aware deferral rule, with an empirical check of whether bothcells samples concentrate in the deferred region.

These are methodology contributions defensible on the framework alone. The empirical claims (does the ensemble beat the gradient-boost baseline, does TTA recover ECE) require the official labels and the planned experiments.

---

## Quick start

```bash
git clone https://github.com/asfandyar-prog/ps3c-robust-inference.git
cd ps3c-robust-inference

uv venv --python 3.11
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\Activate.ps1        # Windows PowerShell

uv pip install -e ".[dev]"
pytest -v
```

To reproduce the seven-team baseline and Stage-2 analysis (requires the team
probability files and the annotated label files):

```bash
python scripts/run_01_verify.py         --data-dir <team-data> --labels-dir <labels>   # Table 2, 14/14 cells
python scripts/run_02_baselines.py      --data-dir <team-data> --labels-dir <labels> --out-dir results
python scripts/run_03_weight_generator.py --data-dir <team-data> --labels-dir <labels> --out-dir results --n-clusters 8 --epochs 100
python scripts/run_05_bbse.py           --data-dir <team-data> --labels-dir <labels> --out-dir results
```
See [`results/README.md`](results/README.md) for which script produces which number.

---

## Repository layout

```
ps3c-robust-inference/
├── configs/                 # YAML configs, one per stage
├── scripts/                 # reproduce_baseline, run_tta, train_ensemble, calibrate_conformal
├── src/ps3c_robust/
│   ├── baseline/            # team model loaders (pending weights)
│   ├── tta/                 # Stage 1: LayerNormTTA, TENT, HybridTTA
│   ├── ensemble/            # Stage 2: AdaptiveEnsemble attention head
│   ├── selective/           # Stage 3: ConformalPredictor with deferral
│   ├── data/                # team probability loaders + alignment
│   ├── eval/                # macro_f1, expected_calibration_error, coverage, selective_risk
│   └── utils/               # seeding, logging
├── tests/                   # pytest
├── docs/                    # design notes
├── data/, weights/, results/  # gitignored
└── notebooks/               # exploration (placeholder)
```

---

## BSc thesis connection

This project builds on the framework from:

> **Predictive Self-Supervised Vision Transformers under Test-Time Distribution Shifts with Lightweight TTA**
> Asfand Yar, BSc Thesis, University of Debrecen, 2026.
> Supervisor: Dr. Bogacsovics Gergo. External supervisor: Sergio Correa (BMW Q-Lab Debrecen).

The thesis develops LayerNorm-only entropy minimization as a TTA mechanism on MedMNIST benchmarks. The implementation in `src/ps3c_robust/tta/layernorm_tta.py` is a faithful port of the thesis `TTAWrapper`: deepcopy for episodic restoration, Adam at 1e-4 over LayerNorm scale and shift only, optimizer reinstantiation on reset, and optional entropy-threshold gating. This project applies that mechanism to a real clinical pipeline with a naturally occurring preprocessing shift, across a heterogeneous ensemble.

---

## Roadmap

- [x] Repository scaffold and uv environment
- [x] Stage 1 (LayerNormTTA) ported from BSc thesis
- [x] Stage 2 sample-adaptive ensemble head
- [x] Stage 3 split-conformal predictor with deferral
- [x] Metrics: macro F1, ECE, coverage, selective risk
- [x] Seven-team probability ingest and alignment
- [x] Obtain official APACC ground-truth labels
- [x] Finalize baseline reproduction against official labels (Table 2, 14/14 cells)
- [x] Stage 2 evaluation: ensemble baselines + Weight Generator vs rank averaging
- [x] Label-shift (BBSE) analysis: diagnostic, correction, matched-correction ablation
- [x] Stage 3 run: conformal coverage (within-test vs cross-split)
- [ ] Stage 1 evaluation on real data: F1 and ECE per team, with and without TTA (blocked on APACC + HPC)
- [ ] Stage 1 ablation: LayerNorm vs BatchNorm in hybrid models
- [ ] Journal / workshop paper draft

---

## Authors

**Developer:** Asfand Yar ([@asfandyar-prog](https://github.com/asfandyar-prog))
**Supervisor:** Prof. Balazs Harangi, Deputy Dean, Department of Data Science and Visualization, University of Debrecen.

---

## License

MIT (provisional, to be confirmed with supervisor before any formal release). The APACC dataset is distributed under CC-BY 4.0 by its original authors.
