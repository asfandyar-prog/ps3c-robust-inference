# Data

This directory holds APACC and any derived artifacts. Everything inside `raw/`,
`processed/`, and `model_outputs/` is gitignored.

## APACC

- Source: <https://osf.io/fp2xe/>
- License: CC-BY 4.0
- Total: 103,675 annotated cervical cell images
- Split: 87 train smears / 20 test smears
- Class counts: 34,721 healthy / 2,942 unhealthy / 62,074 rubbish / 3,884 bothcells

## Expected layout

```
data/
├── raw/
│   └── apacc/
│       ├── train/
│       │   ├── healthy/
│       │   ├── unhealthy/
│       │   ├── rubbish/
│       │   └── bothcells/
│       ├── test/
│       │   └── ... (same four subfolders)
│       ├── eval/
│       │   └── ... (the held-out evaluation set used in the challenge)
│       └── metadata.csv      # per-image: smear_id, split, label, preprocessing flags
├── processed/                # any cached tensors, embeddings, splits
└── model_outputs/            # per-team softmax probabilities on test + eval
    ├── jng_test_probs.npy    # shape: (N_test, 4)  order: [healthy, unhealthy, rubbish, bothcells]
    ├── jng_eval_probs.npy
    ├── ymg_test_probs.npy
    ├── ymg_eval_probs.npy
    ├── ngu_test_probs.npy
    ├── ngu_eval_probs.npy
    ├── gup_test_probs.npy
    ├── gup_eval_probs.npy
    ├── wan_test_probs.npy
    ├── wan_eval_probs.npy
    ├── dpz_test_probs.npy
    ├── dpz_eval_probs.npy
    ├── cha_test_probs.npy
    ├── cha_eval_probs.npy
    └── manifest.csv          # image_id ↔ row index mapping (must be consistent across teams)
```

## Note on the manifest

The probability arrays from the seven teams must share a single canonical row
order for both `test` and `eval`. `manifest.csv` is the source of truth — the
loader in `src/ps3c_robust/data/apacc.py` will assert this consistency on load.
