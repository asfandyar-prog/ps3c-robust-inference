<div align="center">

# PS3C Robust Inference

**Shift-robust ensembling and conformal selective prediction for cervical cell classification.**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](#status)
[![uv](https://img.shields.io/badge/managed%20with-uv-orange)](https://docs.astral.sh/uv/)

*A research project extending the PS3C: Pap Smear Cell Classification Challenge (Kupas, Harangi et al.).*

</div>

---

## Status

Active research project under Prof. Balazs Harangi (University of Debrecen). The
paper is a **two-stage framework** — shift-robust ensembling (Stage 2) and
conformal selective prediction (Stage 3). The official ground-truth labels are in
hand; all seven teams' probability outputs are ingested and aligned, paper Table 2
is reproduced exactly, and Stage 2 ensembling plus a label-shift (BBSE) analysis
are complete.

> The stages keep the numbers **2** and **3** from the project's original
> three-stage design, so the labels stay aligned with the `run_0N` scripts, the
> results files, and the code. Architecture-aware TTA (the former "Stage 1") is
> out of scope — see [Out of scope / Future work](#out-of-scope--future-work).

| Component | Status | Notes |
|---|---|---|
| Repository scaffold | Complete | uv + Python 3.11 |
| Official ground-truth labels | **Obtained** | annotated test + eval sets |
| Seven-team probability ingest | **Complete** | all 7 teams aligned by image name, both splits |
| Baseline reproduction (paper Table 2) | **Reproduced** | 14/14 cells to 4 decimals (`run_01_verify.py`) |
| Stage 2 — shift-robust ensembling + Weight Generator | **Complete** | see Results |
| Label-shift (BBSE) analysis | **Complete** | `run_05` / `run_06`, see Results |
| Stage 3 — conformal selective prediction | **Complete** | `run_04_conformal.py`, see Results |
| Journal / workshop paper | In progress | Target: Medical Image Analysis / ISBI 2027 |

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

The strongest individual models and the meta-learner ensemble lose only a few points, but every system loses something, and no team's reported method actively addresses the shift. This work characterizes the shift (it turns out to be a **label/prior shift**, not general covariate shift) and asks which way of *combining* the teams survives it — the focus of Stage 2 — with Stage 3 adding formal coverage guarantees on top. (Architecture-aware test-time adaptation, one considered remedy, is out of scope here — see below.)

### Issue 2 — Calibration is unreported

The PS3C paper reports accuracy and macro-F1 but not calibration, and offers no notion of selective or deferred prediction. This matters clinically: a 92% F1 ensemble that is overconfident on its errors is harder to deploy safely than one that can abstain when unsure. Stage 3 addresses this directly — split-conformal prediction turns the ensemble's probabilities into prediction sets with a formal marginal-coverage guarantee and defers ambiguous cases (including the "bothcells" class the teams ignored).

(A separate hypothesis from the developer's BSc thesis — that LayerNorm-only entropy minimization recovers calibration under domain shift — motivated the architecture-aware TTA code, which is retained as future work and out of scope for this paper; see below.)

---

## The two-stage framework

> Stages are numbered **2** and **3**, retained from the project's original
> three-stage design so the labels stay aligned with the `run_0N` scripts, the
> results files, and the code (e.g. `conformal.py` refers to "Stage 3"). Stage 1
> (architecture-aware TTA) is out of scope — see
> [Out of scope / Future work](#out-of-scope--future-work).

```
      seven team probability vectors  (per image, 3 classes)
                   |
                   v
   STAGE 2  Shift-Robust Ensembling
   --------------------------------------------------
     parameter-free rules (rank averaging, ...) vs
     learned combiners (gradient boosting, attention
     heads); BBSE analysis of the test->eval shift
                   |
                   v   fused probabilities
   STAGE 3  Conformal Selective Prediction
   --------------------------------------------------
     split-conformal -> marginal coverage guarantee
     ambiguous cases (incl. bothcells) -> defer
                   |
                   v
            Predict OR Defer
```

### Stage 2 — Shift-Robust Ensembling

The question is which way of combining the seven teams survives the test→eval shift. We compare parameter-free rules (rank averaging, simple/geometric mean, hard voting) against learned combiners (gradient boosting, and two attention heads — a cluster-routing Weight Generator and a sample-adaptive AdaptiveEnsemble), and add a BBSE label-shift analysis. The finding: every *learned* combiner collapses on the shifted eval set, while **parameter-free rank averaging is shift-invariant by construction and remains the champion**. See [Results](#results-stage-2--label-shift-analysis).

### Stage 3 — Conformal Selective Prediction

Split-conformal prediction (LAC / APS scoring) turns the ensemble probabilities into prediction sets with a target marginal coverage. The system defers when the prediction set has more than one label (a "bothcells"-in-set rule is available for the 4-class layout; the current pipeline is 3-class). Empirically, within-split conformal meets its coverage guarantee, and the same calibration applied across the test→eval shift under-covers — an independent, quantitative signature of the shift. See [Results](#results-stage-2--label-shift-analysis).

---

## What is and is not novel

Stated plainly, because it shapes how the work should be reviewed.

**Reused from prior work:**
- APS / LAC split-conformal scoring (Romano et al., 2020; Sadinle et al., 2019)
- Black Box Shift Estimation for label shift (Lipton et al., 2018)

**Contributed here:**
- A quantitative characterization of the PS3C test→eval shift as a **label (prior) shift** (unhealthy prior up 5.98×), verified against ground-truth eval labels, and the observation that **parameter-free rank averaging is robust to it while every learned combiner collapses**.
- A BBSE label-shift analysis on the ensemble, with a matched-correction ablation that isolates a train/serve mismatch from a genuine failure of correction.
- Conformal selective prediction on this benchmark, using cross-split under-coverage as an independent signature of the shift, plus a bothcells-aware deferral rule.

These claims are backed by results computed on the official labels; every number is cited to a file in [`results/README.md`](results/README.md).

---

## Out of scope / Future work

**Architecture-aware test-time adaptation (the original "Stage 1").** The TTA code is implemented and retained (`src/ps3c_robust/tta/`: `LayerNormTTA` for ViTs, `TENT` for CNNs, `HybridTTA` for hybrids, dispatched per architecture) but **not evaluated**, and it is out of scope for this paper, for two concrete reasons:

- **No team checkpoints exist.** All seven PS3C submissions are code-only; without trained weights there is no model to adapt at test time, so TTA would require training proxy models from scratch on APACC (HPC-scale, out of scope here).
- **Mechanism mismatch.** Entropy-minimization TTA (TENT / LayerNorm) targets *covariate* shift, whereas the shift identified here is a *label (prior)* shift (unhealthy prevalence up 5.98×). Its expected benefit is therefore limited; label-shift correction (BBSE, evaluated in Stage 2) is the more appropriate tool.

The implementation is kept in the repository as a basis for future work — not deleted.

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
├── configs/                 # YAML configs (tta.yaml retained but out of scope)
├── scripts/                 # active: run_01_verify … run_06_bbse_matched; plus legacy scaffold stubs
├── src/ps3c_robust/
│   ├── baseline/            # team model loaders (pending weights; TTA scaffold)
│   ├── tta/                 # (out of scope) LayerNormTTA, TENT, HybridTTA — retained, not evaluated
│   ├── ensemble/            # Stage 2: AdaptiveEnsemble attention head
│   ├── adapt/               # Stage 2: BBSE label-shift correction
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

The **out-of-scope** TTA component (retained as future work) originates in the developer's BSc thesis:

> **Predictive Self-Supervised Vision Transformers under Test-Time Distribution Shifts with Lightweight TTA**
> Asfand Yar, BSc Thesis, University of Debrecen, 2026.
> Supervisor: Dr. Bogacsovics Gergo. External supervisor: Sergio Correa (BMW Q-Lab Debrecen).

The thesis develops LayerNorm-only entropy minimization as a TTA mechanism on MedMNIST benchmarks. The implementation in `src/ps3c_robust/tta/layernorm_tta.py` is a faithful port of the thesis `TTAWrapper`: deepcopy for episodic restoration, Adam at 1e-4 over LayerNorm scale and shift only, optimizer reinstantiation on reset, and optional entropy-threshold gating. Applying it to this ensemble is left to future work (see [Out of scope](#out-of-scope--future-work)); it is not part of the two-stage pipeline evaluated here.

---

## Roadmap

- [x] Repository scaffold and uv environment
- [x] Stage 2 & 3 frameworks implemented (ensembling, split-conformal + deferral)
- [x] Metrics: macro F1, ECE, coverage, selective risk
- [x] Seven-team probability ingest and alignment
- [x] Obtain official APACC ground-truth labels
- [x] Finalize baseline reproduction against official labels (Table 2, 14/14 cells)
- [x] Stage 2 evaluation: ensemble baselines + Weight Generator vs rank averaging
- [x] Label-shift (BBSE) analysis: diagnostic, correction, matched-correction ablation
- [x] Stage 3 run: conformal coverage (within-test vs cross-split)
- [ ] Journal / workshop paper draft
- [ ] *(future work)* architecture-aware TTA evaluation — blocked on team checkpoints; see [Out of scope](#out-of-scope--future-work)

---

## Authors

**Developer:** Asfand Yar ([@asfandyar-prog](https://github.com/asfandyar-prog))
**Supervisor:** Prof. Balazs Harangi, Deputy Dean, Department of Data Science and Visualization, University of Debrecen.

---

## License

MIT (provisional, to be confirmed with supervisor before any formal release). The APACC dataset is distributed under CC-BY 4.0 by its original authors.
