# PS3C Robust Inference

A three-stage framework that addresses the preprocessing-induced distribution shift
in cervical cell classification, extending the **PS3C: Pap Smear Cell Classification
Challenge** (Kupas, Harangi et al.).

> **Status:** scaffolding complete, awaiting model weights and per-team probability
> outputs from Prof. Balazs Harangi (Univ. of Debrecen) before reproducing the
> baseline.

## Background

The PS3C challenge asked seven teams to classify cervical cells from the APACC
dataset (103,675 images) into four categories — **healthy**, **unhealthy**,
**rubbish**, **bothcells** — although every team ignored *bothcells* in practice.

The challenge surfaced a striking failure mode that nobody addressed: a
preprocessing-induced distribution shift between the test and evaluation sets
caused large F1 collapses across all submissions.

| Team       | Test F1 | Eval F1 | Drop   |
|------------|---------|---------|--------|
| DPZ        | 0.8622  | 0.7092  | -17.7% |
| GUP        | 0.8604  | 0.7211  | -15.9% |
| Best ensemble (Gradient Boost) | 0.9517 | 0.9245 | -2.9% |

Closing this gap is the contribution of this paper.

## The Three-Stage Framework

### Stage 1 — Architecture-Aware Test-Time Adaptation
Different model families need different adaptation surfaces.

| Model family | Mechanism                          | Applies to              |
|--------------|------------------------------------|-------------------------|
| ViT          | LayerNorm scale/shift entropy TTA  | JNG, YMG, NGU           |
| CNN          | TENT (BatchNorm statistics)        | GUP, WAN                |
| Hybrid       | Both, simultaneously               | DPZ, CHA                |

The LayerNorm TTA variant comes from the developer's BSc thesis; combining it
with TENT inside a heterogeneous ensemble is novel.

### Stage 2 — Sample-Adaptive Ensemble
A lightweight attention head that produces **per-sample** weights over the seven
model outputs, replacing the static gradient-boost weights of the original paper.
Different cell types should trust different models.

### Stage 3 — Selective Prediction
Conformal prediction with formal coverage guarantees (target: 95%) lets the
system defer ambiguous cases — including the bothcells samples that no team
modelled directly — to a cytologist.

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

- [x] Repo scaffold + uv environment
- [ ] Receive model weights + probability outputs
- [ ] Reproduce per-team test/eval F1 numbers
- [ ] Reproduce gradient-boost ensemble baseline
- [ ] Implement LayerNorm TTA on ViT models (port from BSc thesis)
- [ ] Implement TENT on CNN models
- [ ] Implement hybrid TTA — ablation: LayerNorm vs BatchNorm contribution
- [ ] Implement sample-adaptive ensemble attention
- [ ] Implement conformal selective prediction with bothcells deferral analysis
- [ ] Full pipeline evaluation
- [ ] Paper draft (target: Medical Image Analysis / ISBI 2027)

## Authors

- **Developer:** Asfand Yar — yarasfand886@gmail.com
- **Supervisor:** Prof. Balazs Harangi, Deputy Dean, Dept. of Data Science and
  Visualization, University of Debrecen

## License

MIT (provisional — confirm with supervisor before public release).
