# PS3C Robust Inference

A three-stage framework that addresses the preprocessing-induced distribution shift
in cervical cell classification, extending the **PS3C: Pap Smear Cell Classification
Challenge** (Kupas, Harangi et al.).

> **Status:** scaffolding complete, awaiting model weights and per-team probability
> outputs from Prof. Balazs Harangi (Univ. of Debrecen) before reproducing the
> baseline.

## Background

The PS3C challenge asked seven teams to classify cervical cells from the APACC
dataset (103,675 images) into three clinically relevant categories — **healthy**,
**unhealthy**, **rubbish** — with a fourth ambiguous category, **bothcells**,
that all teams handled differently or ignored.

The challenge surfaced two deployment-relevant failure modes that the original
paper documented but did **not** address.

### Failure mode 1 — preprocessing-induced shift

The hidden evaluation set used a slightly different preprocessing pipeline than
the test set: cell crops were normalized to 224×224 with white-pixel padding,
where the public test set was not. Every team's macro F1 dropped on the
evaluation set, with a wide spread depending on whether the team happened to
match the eval-side preprocessing during training.

| Team / system                   | Test F1 | Eval F1 | Δ (abs) |
|---------------------------------|---------|---------|---------|
| DPZ (CNN+ViT hybrid)            | 0.8622  | 0.7092  | -0.153  |
| GUP (ResNet ensemble)           | 0.8604  | 0.7211  | -0.139  |
| YMG (MaxViT, padded already)    | 0.8680  | 0.7858  | -0.082  |
| JNG (foundation models)         | 0.8702  | 0.8176  | -0.053  |
| **Best ensemble (Gradient Boost)** | **0.9517** | **0.9245** | **-0.027** |

The strongest individual models and the gradient-boost ensemble lose only a few
points, but uniformly — and no team's reported method actively addresses the
shift. **Stage 1 of this work** (architecture-aware TTA) closes this remaining
gap without retraining and without target-domain labels.

### Failure mode 2 — calibration is unreported

The PS3C paper reports accuracy and macro F1 but **does not report calibration**.
This matters clinically: a 92% F1 ensemble that is overconfident on its 8% of
errors is harder to deploy than a 92% F1 ensemble that knows when it might be
wrong.

This omission is the more interesting gap. The developer's BSc thesis showed
that supervised ViTs under medical domain shift exhibit a 64× rise in Expected
Calibration Error (0.014 → 0.889 on PathMNIST → DermaMNIST), and that
LayerNorm-only entropy minimization recovers calibration without any target
labels. **Stage 1 is expected to recover ECE on the PS3C eval set, and Stage 3
turns the resulting calibrated probabilities into formal coverage guarantees.**

## The Three-Stage Framework

### Stage 1 — Architecture-Aware Test-Time Adaptation
Different model families need different adaptation surfaces.

| Model family | Mechanism                          | Applies to              |
|--------------|------------------------------------|-------------------------|
| ViT          | LayerNorm scale/shift entropy TTA  | JNG, YMG, NGU           |
| CNN          | TENT (BatchNorm statistics)        | GUP, WAN                |
| Hybrid       | Both, simultaneously               | DPZ, CHA                |

The LayerNorm TTA variant is ported directly from the developer's BSc thesis
implementation. Combining it with TENT inside a heterogeneous seven-model
ensemble — and ablating which adaptation surface matters more for hybrid
models — is novel to this work.

### Stage 2 — Sample-Adaptive Ensemble
A lightweight attention head that produces **per-sample** weights over the seven
team outputs, replacing the static gradient-boost weights of the original paper.
Different cell types should trust different models.

### Stage 3 — Selective Prediction
Split-conformal prediction with a target marginal coverage of 95% lets the
system defer ambiguous cases — including the bothcells samples that no team
modelled cleanly — to a cytologist. The same entropy threshold that gates TTA
adaptation in Stage 1 also drives the Stage 3 deferral decision, giving the
pipeline a single, interpretable confidence axis.

## Connection to the BSc Thesis

This work clinically grounds the framework introduced in:

> *Predictive Self-Supervised Vision Transformers under Test-Time Distribution
> Shifts with Lightweight TTA* — benchmarked on PathMNIST, DermaMNIST,
> RetinaMNIST.
>
> Supervisor: Dr. Bogacsovics Gergő (UniDeb).
> External supervisor: Sergio Correa (BMW Q-Lab Debrecen).

## Project Structure

```
ps3c-robust-inference/
├── configs/                 # YAML configs per stage
├── data/                    # APACC raw + processed (gitignored)
├── weights/                 # team model weights from Prof. Harangi (gitignored)
├── results/                 # experiment outputs (gitignored)
├── notebooks/               # exploration + ablations
├── scripts/                 # entry points (one per pipeline stage)
├── src/ps3c_robust/
│   ├── baseline/            # Stage 0: load + reproduce 7 team models
│   ├── tta/                 # Stage 1: LayerNorm TTA, TENT, hybrid
│   ├── ensemble/            # Stage 2: sample-adaptive attention
│   ├── selective/           # Stage 3: conformal prediction + deferral
│   ├── data/                # APACC dataset loader
│   ├── eval/                # F1, coverage, distribution-shift metrics
│   └── utils/               # logging, seeding
├── tests/                   # pytest
└── docs/                    # design notes
```

## Setup (Windows, VS Code)

This project uses [`uv`](https://docs.astral.sh/uv/) and Python 3.11.

```powershell
# 1. Install uv if you don't have it
# https://docs.astral.sh/uv/getting-started/installation/

# 2. Clone the repo
git clone https://github.com/asfandyar-prog/ps3c-robust-inference.git
cd ps3c-robust-inference

# 3. Create the venv with Python 3.11
uv venv --python 3.11

# 4. Activate it (PowerShell)
.venv\Scripts\Activate.ps1

# 5. Install the project in editable mode with dev tools
uv pip install -e ".[dev]"
```

If you have an NVIDIA GPU and want CUDA-enabled PyTorch, install it explicitly
*before* the editable install (uv resolves cleanly afterwards):

```powershell
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install -e ".[dev]"
```

VS Code: open the folder, `Ctrl+Shift+P` → "Python: Select Interpreter" →
choose `.venv\Scripts\python.exe`.

## Data

APACC is publicly available at [osf.io/fp2xe](https://osf.io/fp2xe/) under
CC-BY 4.0. See `data/README.md` for the expected layout once downloaded.

## Weights

The seven team weight files and their per-image probability outputs on the test
and evaluation sets are pending from Prof. Harangi. See `weights/README.md` for
the expected layout — the pipeline will validate against this layout on first
run.

## Roadmap

Code milestones:

- [x] Repo scaffold + uv environment
- [x] Stage 1 — `LayerNormTTA` ported from BSc thesis (with deepcopy snapshot,
      entropy-threshold gating, optimizer reset)
- [x] Stage 2 — sample-adaptive ensemble head (working module)
- [x] Stage 3 — split-conformal predictor with deferral logic (working module)
- [x] Metrics — macro F1, ECE (15-bin equal-width), coverage, selective risk
- [ ] Receive model weights + per-team probability outputs from Prof. Harangi
- [ ] Reproduce per-team test/eval F1 and ECE numbers
- [ ] Reproduce gradient-boost ensemble baseline (with ECE)
- [ ] TTA ablation on hybrid models — LayerNorm vs BatchNorm contribution
- [ ] Bothcells deferral analysis (Stage 3)

Paper milestones:

- [ ] Workshop paper (target: ISBI 2026 workshop or MICCAI 2026 workshop)
- [ ] Full paper (target: Medical Image Analysis / ISBI 2027)

Headline experimental questions:

1. Does Stage 1 close the residual eval-set F1 gap left by the gradient-boost
   baseline, without retraining?
2. Does Stage 1 also recover ECE — the metric the original challenge did not
   report?
3. Does the sample-adaptive ensemble (Stage 2) outperform gradient-boost
   stacking on both F1 and ECE?
4. Do bothcells samples cluster in the deferred region of the conformal
   predictor (Stage 3), confirming that the system can flag the failure mode
   no team modelled?

## Authors

- **Developer:** Asfand Yar — yarasfand886@gmail.com
- **Supervisor:** Prof. Balazs Harangi, Deputy Dean, Dept. of Data Science and
  Visualization, University of Debrecen

## License

MIT (provisional — confirm with supervisor before public release).
