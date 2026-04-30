# Notebooks

Add notebooks here as the project progresses. Suggested ordering matches
the four pipeline stages:

1. `01_data_exploration.ipynb` — APACC class balance, smear-level stats,
   eyeballing the test/eval shift visually.
2. `02_baseline_reproduction.ipynb` — confirm we hit the published per-team
   and ensemble F1 numbers.
3. `03_tta_ablations.ipynb` — Stage 1 ablations (per family, per surface).
4. `04_ensemble_analysis.ipynb` — visualize per-sample team-attention
   weights; do they correlate with cell type?
5. `05_selective_prediction.ipynb` — coverage curves, deferral histograms,
   bothcells routing analysis.

Run notebooks with the project's venv (`.venv\Scripts\python.exe` on
Windows). After `uv pip install -e ".[dev]"` the kernel is available as
`python` — register it explicitly with:

```powershell
python -m ipykernel install --user --name ps3c-robust --display-name "ps3c-robust"
```
