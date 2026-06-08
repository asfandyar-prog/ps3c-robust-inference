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

This is an active research project under the supervision of Prof. Balazs Harangi (University of Debrecen). The framework is implemented and the data ingest works on real team outputs. Baseline reproduction is in progress, pending the official ground-truth labels.

| Component | Status | Notes |
|---|---|---|
| Repository scaffold | Complete | uv + Python 3.11 |
| Stage 1 framework (TTA) | Implemented | Ported from BSc thesis |
| Stage 2 framework (ensemble) | Implemented | Sample-adaptive attention head |
| Stage 3 framework (conformal) | Implemented | Split-conformal with deferral |
| Metrics (F1, ECE, coverage, selective risk) | Implemented | ECE is the calibration metric the original challenge did not report |
| Six-team probability ingest | Working | Aligns all six teams by image name across both splits |
| Baseline reproduction | In progress | Runs end to end; numbers not yet final (see note below) |
| Official ground-truth labels | Pending | Needed to finalize the baseline |
| Stage 1, 2, 3 experiments on real data | Planned | After labels are confirmed |
| Workshop paper | Planned | Target: a 2027 venue (MIDL / ISBI) |

> **Why the baseline is not final yet.** The six teams' probability files each contain their own `label` column, but these disagree in a few hundred cases (175 on test, 496 on evaluation), so they are not an authoritative reference. Until the official APACC label file is available, reproduced F1 numbers will not exactly match the published paper. The ingest and the pipeline are complete; only the answer key is missing.

> **Data availability.** Six of the seven challenge teams' outputs are in hand (YMG, JNG, CHA, GUP, DPZ, WAN). The seventh (NGU) was not among the shared files and its code ships without trained weights, so the current work proceeds with six teams.

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

To reproduce the six-team baseline (requires the team probability files arranged as the organizer share):

```bash
python scripts/reproduce_baseline.py --data-dir /path/to/ps3c-team-data
```

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
- [x] Six-team probability ingest and alignment
- [ ] Obtain official APACC ground-truth labels
- [ ] Finalize baseline reproduction against official labels
- [ ] Stage 1 evaluation: F1 and ECE per team, with and without TTA
- [ ] Stage 1 ablation: LayerNorm vs BatchNorm in hybrid models
- [ ] Stage 2 evaluation: adaptive ensemble vs gradient-boost stacking
- [ ] Stage 3 evaluation: coverage, deferral rate, bothcells routing
- [ ] Workshop paper draft

---

## Authors

**Developer:** Asfand Yar ([@asfandyar-prog](https://github.com/asfandyar-prog))
**Supervisor:** Prof. Balazs Harangi, Deputy Dean, Department of Data Science and Visualization, University of Debrecen.

---

## License

MIT (provisional, to be confirmed with supervisor before any formal release). The APACC dataset is distributed under CC-BY 4.0 by its original authors.
