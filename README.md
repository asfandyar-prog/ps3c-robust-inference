<div align="center">

# PS3C Robust Inference

**Architecture-aware test-time adaptation, sample-adaptive ensembling, and conformal selective prediction for cervical cell classification.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-4%2F4%20passing-brightgreen.svg)](#status)
[![uv](https://img.shields.io/badge/managed%20with-uv-orange)](https://docs.astral.sh/uv/)

*A research project extending the [PS3C: Pap Smear Cell Classification Challenge](https://arxiv.org/abs/) (Kupas, Harangi et al.).*

</div>

---

## Status

This is an active research project. The methodology and implementation are in place; experimental validation is pending model weights from the original challenge organizers.

| Component | Status | Notes |
|---|---|---|
| Repository scaffold | ✅ Complete | `uv` + Python 3.11, MIT license |
| Stage 1 — Architecture-Aware TTA | ✅ Implemented | Ported from [BSc thesis](#bsc-thesis-connection) |
| Stage 2 — Sample-Adaptive Ensemble | ✅ Implemented | Lightweight attention head |
| Stage 3 — Conformal Selective Prediction | ✅ Implemented | Split-conformal with deferral |
| Metrics — F1, ECE, Coverage, Selective Risk | ✅ Implemented | Includes the calibration metric the original challenge omitted |
| Test suite | ✅ 4/4 passing | `pytest -v` |
| **Model weights from challenge organizers** | 🟡 **Pending** | Required for baseline reproduction |
| Baseline reproduction | ⬜ Blocked on weights | Target: match published F1 numbers |
| TTA experimental results | ⬜ Blocked on weights | Stage 1 validation |
| Sample-adaptive ensemble training | ⬜ Blocked on weights | Stage 2 validation |
| Conformal calibration on real data | ⬜ Blocked on weights | Stage 3 validation |
| Workshop paper draft | ⬜ Planned | Target: ISBI 2026 / MICCAI 2026 satellite workshop |

> **What "blocked on weights" means.** The seven team checkpoints and per-image probability outputs from the original challenge are needed to reproduce the baseline and run end-to-end experiments. Until those arrive from Prof. Harangi, the pipeline runs on a synthetic harness only — see [Roadmap](#roadmap) for what we're doing in the meantime.

---

## The problem this work addresses

The PS3C challenge benchmarked seven deep-learning teams on cervical cell classification using the [APACC dataset](https://osf.io/fp2xe/) (103,675 images across 107 patients). The reported results in the proceedings paper revealed two deployment-relevant failure modes that the original work documented but did not address.

### Failure mode 1 — Preprocessing-induced distribution shift

The hidden evaluation set used a slightly different preprocessing pipeline than the test set: cell crops were normalized to 224×224 with white-pixel padding. Every team's macro-F1 dropped on the evaluation set, with the size of the drop depending heavily on whether each team happened to match the eval-side preprocessing during training.

| Team / system | Architecture family | Test F1 | Eval F1 | Δ (absolute) |
|---|---|---|---|---|
| DPZ | Hybrid (CNN+ViT) | 0.8622 | 0.7092 | **−0.153** |
| GUP | CNN ensemble | 0.8604 | 0.7211 | **−0.139** |
| YMG | MaxViT (padded already) | 0.8680 | 0.7858 | −0.082 |
| JNG | Foundation models + LoRA | 0.8702 | 0.8176 | −0.053 |
| **Best ensemble** (Gradient Boost stacking) | Meta-learner | **0.9517** | **0.9245** | **−0.027** |

The strongest individual models and the meta-learner ensemble lose only a few points. But every system loses points, and **no team's reported method actively addresses the shift.** Closing this residual gap *without retraining* and *without target-domain labels* is the goal of Stage 1.

### Failure mode 2 — Calibration is unreported

The PS3C paper reports accuracy and macro-F1 but does not report calibration. This matters clinically: a 92% F1 ensemble that is overconfident on its 8% of errors is harder to deploy safely than a 92% F1 ensemble whose probabilities reflect its actual reliability.

The connection to my BSc thesis is direct: that work measured a 64× rise in Expected Calibration Error (0.014 → 0.889) for supervised ViTs under medical domain shift, and showed that LayerNorm-only entropy minimization recovers calibration without target-domain labels. The hypothesis being tested in this project is that the same mechanism, generalized to a heterogeneous seven-model ensemble, recovers calibration on the PS3C eval set — and that Stage 3 then turns the resulting calibrated probabilities into formal coverage guarantees.

---

## The three-stage framework

```
┌─────────────────────┐
│   APACC test image  │
└──────────┬──────────┘
           │
           ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — Architecture-Aware Test-Time Adaptation                 │
│                                                                    │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│   │ ViT models   │   │ CNN models   │   │ Hybrid       │           │
│   │ JNG/YMG/NGU  │   │ GUP/WAN      │   │ DPZ/CHA      │           │
│   │              │   │              │   │              │           │
│   │ LayerNorm    │   │ TENT         │   │ Both         │           │
│   │ γ/β entropy  │   │ BatchNorm    │   │ surfaces     │           │
│   │ minimization │   │ statistics   │   │ jointly      │           │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────┘           │
│          └──────────────────┼──────────────────┘                   │
└─────────────────────────────┼──────────────────────────────────────┘
                              │ 7 adapted probability vectors
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — Sample-Adaptive Ensemble                                │
│                                                                    │
│  Lightweight attention head produces per-sample weights over       │
│  the 7 team outputs (replacing static gradient-boost weights).     │
│  Different cell types should trust different models.               │
└─────────────────────────────┬──────────────────────────────────────┘
                              │ fused probabilities
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — Conformal Selective Prediction                          │
│                                                                    │
│  Split-conformal calibration → 95% marginal coverage guarantee.    │
│  Bothcells samples (no team modeled cleanly) cluster in deferred   │
│  region and route to a cytologist.                                 │
└─────────────────────────────┬──────────────────────────────────────┘
                              │
                              ▼
                  ┌─────────────────────┐
                  │ Predict OR Defer    │
                  └─────────────────────┘
```

### Stage 1 — Architecture-Aware Test-Time Adaptation

Different model families need different adaptation surfaces. The original TENT (Wang et al., ICLR 2021) targets BatchNorm and therefore does not apply to Vision Transformers. Conversely, LayerNorm-based TTA does not apply to pure CNNs. PS3C contains both, plus hybrids — so adaptation must be dispatched per architecture.

| Family | Mechanism | Targets | Applies to |
|---|---|---|---|
| ViT | Entropy minimization on LayerNorm γ, β | ~18K params on ViT-B/16 | JNG, YMG, NGU |
| CNN | TENT — BatchNorm γ, β + recomputed running stats | per architecture | GUP, WAN |
| Hybrid | Both surfaces simultaneously | combined | DPZ, CHA |

The LayerNorm TTA implementation is ported directly from my BSc thesis ([details below](#bsc-thesis-connection)). What is novel to this work is **the dispatch pattern itself** — applying the right adaptation surface to each model in a heterogeneous ensemble, and ablating which surface matters more in hybrid architectures.

### Stage 2 — Sample-Adaptive Ensemble

A lightweight attention head produces per-sample weights over the seven team outputs, replacing the original paper's static gradient-boost weights.

**Inputs per sample, per team:**
- Softmax probabilities (4 dims)
- Confidence statistics: max-prob, entropy, top-1/top-2 margin (3 dims)

**Output:** fused 4-class probability vector + per-sample interpretability weights showing which team was trusted for each prediction.

The hypothesis: different cell types have different optimal models, and a static ensemble cannot exploit this. Whether this holds is one of the experimental questions the project will answer.

### Stage 3 — Conformal Selective Prediction

Split-conformal prediction (APS / LAC scoring) with a target marginal coverage of 95%. The same entropy threshold that gates Stage 1 adaptation also drives the Stage 3 deferral decision, giving the pipeline a single, interpretable confidence axis.

**Deferral logic:**
- Defer if conformal prediction set has more than one label.
- Defer if `bothcells` is in the set above a configurable probability threshold.

The bothcells category is interesting precisely because no team modeled it cleanly. The expectation — testable once weights arrive — is that bothcells samples cluster in the deferred region, demonstrating that Stage 3 can flag the failure mode the original challenge surfaced but ignored.

---

## What is and is not novel here

I want to be explicit about this because it shapes how the work should be read and reviewed.

**Components reused from prior work:**
- Entropy-minimization TTA loss (Wang et al., TENT, ICLR 2021)
- LayerNorm parameter targeting (TTT++, Liu et al., NeurIPS 2021)
- APS / LAC conformal scoring (Romano et al., 2020; Sadinle et al., 2019)

**What this project contributes:**
- A dispatch pattern that applies the right TTA surface per architecture across a heterogeneous ensemble — including ablation of LayerNorm vs BatchNorm contribution within hybrid models.
- ECE as a first-class metric on the PS3C benchmark, which the original challenge does not report.
- A unified entropy threshold that gates both adaptation (Stage 1) and deferral (Stage 3), giving the pipeline one interpretable confidence dial.
- A bothcells-aware deferral rule with an empirical analysis of whether bothcells samples actually cluster in the deferred region.

These are methodology contributions and can be defended on the framework alone. The empirical claims that *will* require experimental support — does Stage 1 close the F1 gap, does it recover ECE, does the adaptive ensemble beat gradient-boost — depend on weights from the challenge organizers and have not yet been tested.

---

## Quick start

```bash
# Clone
git clone https://github.com/asfandyar-prog/ps3c-robust-inference.git
cd ps3c-robust-inference

# Environment (Python 3.11)
uv venv --python 3.11
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\Activate.ps1        # Windows PowerShell

# Install (editable + dev tools)
uv pip install -e ".[dev]"

# Run the test suite to verify the install
pytest -v
# Expected: 4 passed
```

For NVIDIA GPU + CUDA 12.4, install PyTorch first with the explicit index, then the project:

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[dev]"
```

---

## Repository layout

```
ps3c-robust-inference/
│
├── configs/                 # YAML configs, one per pipeline stage
├── scripts/                 # Entry points: reproduce_baseline, run_tta, train_ensemble, calibrate_conformal
├── src/ps3c_robust/
│   ├── baseline/            # Stage 0: load + reproduce 7 team models (loaders pending weights)
│   ├── tta/                 # Stage 1: LayerNormTTA, TENT, HybridTTA
│   ├── ensemble/            # Stage 2: AdaptiveEnsemble attention head
│   ├── selective/           # Stage 3: ConformalPredictor with deferral
│   ├── data/                # APACC dataset + per-team probability loaders
│   ├── eval/                # macro_f1, expected_calibration_error, coverage, selective_risk
│   └── utils/               # seeding, logging
├── tests/                   # pytest — 4/4 passing
├── docs/                    # design notes (architecture rationale)
├── data/, weights/, results/  # gitignored — populated as artifacts arrive
└── notebooks/               # exploration + ablations (placeholder)
```

The `src/`-style layout is deliberate: it prevents a class of import bugs that affect ML projects with editable installs.

---

## BSc thesis connection

This project clinically grounds the framework introduced in:

> **Predictive Self-Supervised Vision Transformers under Test-Time Distribution Shifts with Lightweight TTA**
> Asfand Yar — BSc Thesis, University of Debrecen, 2026.
> Supervisor: Dr. Bogacsovics Gergő (UniDeb). External supervisor: Sergio Correa (BMW Q-Lab Debrecen).
> Repo: [github.com/asfandyar-prog/JEPA-RobustViT](https://github.com/asfandyar-prog/JEPA-RobustViT)

The thesis develops and evaluates LayerNorm-only entropy minimization as a TTA mechanism on three MedMNIST benchmarks (PathMNIST, DermaMNIST, RetinaMNIST). The implementation in `src/ps3c_robust/tta/layernorm_tta.py` is a faithful port of the canonical thesis `TTAWrapper`, with all design choices preserved:

- `copy.deepcopy` of the model for episodic state restoration
- Adam optimizer (1e-4) over LayerNorm γ/β only
- Optimizer reinstantiation on every reset
- Optional entropy threshold gating

This project takes the same mechanism and applies it to a real clinical pipeline with a *naturally occurring* preprocessing shift, in a *heterogeneous* seven-model ensemble — settings the thesis does not address.

---

## Roadmap

**Code milestones**

- [x] Repository scaffold and uv environment
- [x] Stage 1 — `LayerNormTTA` ported from BSc thesis
- [x] Stage 2 — Sample-adaptive ensemble head
- [x] Stage 3 — Split-conformal predictor with deferral logic
- [x] Metrics — macro F1, ECE, coverage, selective risk
- [ ] Receive model weights and per-team probability outputs from challenge organizers
- [ ] Baseline reproduction (match published F1 numbers)
- [ ] Stage 1 evaluation: F1 and ECE per team, with and without TTA
- [ ] Stage 1 ablation: LayerNorm vs BatchNorm contribution in hybrid models
- [ ] Stage 2 evaluation: adaptive ensemble vs gradient-boost stacking
- [ ] Stage 3 evaluation: coverage, deferral rate, bothcells routing analysis

**Paper milestones**

- [ ] Workshop submission (target: ISBI 2026 satellite or MICCAI 2026 workshop)
- [ ] Full paper (target: Medical Image Analysis or ISBI 2027 main track)

**Headline experimental questions**

1. Does architecture-aware TTA close the residual eval-set F1 gap left by the gradient-boost baseline, without retraining?
2. Does it also recover ECE — the metric the original challenge does not report?
3. Does the sample-adaptive ensemble outperform gradient-boost stacking on both F1 and ECE?
4. Do bothcells samples cluster in the deferred region of the conformal predictor, validating that the system flags the failure mode no team modeled?

---

## Authors

**Developer:** Asfand Yar — [yarasfand886@gmail.com](mailto:yarasfand886@gmail.com) — [@asfandyar-prog](https://github.com/asfandyar-prog)
**Supervisor:** Prof. Balazs Harangi, Deputy Dean, Department of Data Science and Visualization, University of Debrecen.

---

## License

Released under the [MIT License](LICENSE) (provisional — to be confirmed with supervisor before public release). The APACC dataset itself is distributed under CC-BY 4.0 by the original authors.

---

## Citation

This work has not yet been submitted or published. Citation information will be added once the workshop paper is accepted.
