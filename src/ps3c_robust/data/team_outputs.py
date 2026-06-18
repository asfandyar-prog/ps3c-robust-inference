"""Loader for the six PS3C team probability outputs, validated against the
official APACC ground-truth labels.

This module reproduces the published per-team numbers exactly (accuracy and
weighted-F1, all six teams, both splits, to four decimal places). Getting there
required handling several real quirks in the shared files:

* JNG filenames are swapped — "Test-set.csv" contains eval_image_* rows and
  "Evaluation-set.csv" contains test_image_* rows. Split is assigned by the
  image-name prefix, never by the filename.
* Four different column orderings across teams — remapped by header name.
* DPZ is a two-stage cascade (rubbish-vs-not, then a multi-label healthy /
  unhealthy head), so its three columns are NOT a probability distribution and
  do not sum to one. The correct decision rule is:
      if rubbish >= DPZ_RUBBISH_THRESHOLD -> rubbish
      else -> argmax(healthy, unhealthy)
  A threshold of 0.46 reproduces DPZ's published accuracy on both splits.
* YMG's test file has a bothcells column; eval does not. bothcells is dropped
  and the remaining three classes renormalized (for the non-DPZ teams whose
  outputs are genuine softmaxes).

Two metric notes confirmed during validation:
* The paper's per-team "F1-score" column is sklearn WEIGHTED F1 (F1 per class
  weighted by support), not macro F1, despite the text describing macro F1.
* Accuracy matches the published accuracy column exactly.

Canonical class order: [healthy, unhealthy, rubbish].
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

CANON = ["healthy", "unhealthy", "rubbish"]
CANON_IDX = {c: i for i, c in enumerate(CANON)}

# DPZ two-stage cascade threshold (reverse-engineered to match published acc).
DPZ_RUBBISH_THRESHOLD = 0.46

HEADER_ALIASES = {
    "healthy": "healthy", "healthy_prob": "healthy", "prob_healthy": "healthy",
    "unhealthy": "unhealthy", "unhealthy_prob": "unhealthy", "prob_unhealthy": "unhealthy",
    "rubbish": "rubbish", "rubbish_prob": "rubbish", "prob_rubbish": "rubbish",
    "bothcells": "bothcells", "bothcells_prob": "bothcells",
}

TEAMS = ["YMG", "JNG", "NGU", "CHA", "GUP", "DPZ", "WAN"]


def _team_file_map(data_dir: Path) -> dict[str, dict[str, Path]]:
    d = data_dir
    return {
        "YMG": {
            "test": d / "wrapup_UT/wrapup_UT/test_phase_prob.csv",
            "eval": d / "wrapup_UT/wrapup_UT/final_eval_phase_revised_prob.csv",
        },
        "JNG": {  # swap: contents, not filename, decide the split
            "test": d / "jianght_challenge_materials/Evaluation-set.csv",
            "eval": d / "jianght_challenge_materials/Test-set.csv",
        },
        "NGU": {
            "test": d / "NGU/predictions_isbi2025-ps3c-test-dataset.csv",
            "eval": d / "NGU/predictions_isbi2025-ps3c-eval-dataset.csv",
        },
        "CHA": {
            "test": d / "ChatoLina_challenge_materials/isbi2025-ps3c-test-dataset pro Ens.csv",
            "eval": d / "ChatoLina_challenge_materials/isbi2025-ps3c-eval-dataset pro Ens.csv",
        },
        "GUP": {
            "test": d / "Chinmay materials/ISBI PS3C IIT KGP/Test_Set_ProbabilityScores.csv",
            "eval": d / "Chinmay materials/ISBI PS3C IIT KGP/Eval_Set_ProbabilityScores.csv",
        },
        "DPZ": {
            "test": d / "Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_test.csv",
            "eval": d / "Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_eval.csv",
        },
        "WAN": {
            "test": d / "WAN/validation_predictions2.csv",
            "eval": d / "WAN/val_predictions_resnet50.csv",
        },
    }


def _normalize_name(name: str) -> str:
    n = name.strip()
    return n[:-4] if n.lower().endswith(".png") else n


def _read_csv(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Expected team file not found: {path}\n"
            "Check the data directory matches the organizer share layout."
        )
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = [r for r in reader if r and r[0].strip()]
    return header, rows


def team_predictions(data_dir: str | Path, team: str, split: str) -> dict[str, int]:
    """Return {image_name: predicted_class_index} for one team and split.

    Encapsulates the per-team decision rule, including DPZ's cascade.
    """
    data_dir = Path(data_dir)
    path = _team_file_map(data_dir)[team][split]
    header, rows = _read_csv(path)
    low = [h.lower() for h in header]

    out: dict[str, int] = {}

    if team == "DPZ":
        ri, hi, ui = low.index("rubbish"), low.index("healthy"), low.index("unhealthy")
        for row in rows:
            name = _normalize_name(row[0])
            rub, hea, unh = float(row[ri]), float(row[hi]), float(row[ui])
            if rub >= DPZ_RUBBISH_THRESHOLD:
                out[name] = CANON_IDX["rubbish"]
            else:
                out[name] = CANON_IDX["healthy"] if hea >= unh else CANON_IDX["unhealthy"]
        return out

    col_to_class = {i: HEADER_ALIASES[c] for i, c in enumerate(low) if c in HEADER_ALIASES}
    for row in rows:
        name = _normalize_name(row[0])
        vec = np.zeros(3)
        for i, cls in col_to_class.items():
            if cls == "bothcells":
                continue
            try:
                vec[CANON_IDX[cls]] = float(row[i])
            except (ValueError, IndexError):
                pass
        out[name] = int(vec.argmax())
    return out


def load_official_labels(labels_dir: str | Path, split: str) -> dict[str, int]:
    """Read the official APACC ground-truth labels for a split.

    Expects files named:
        isbi2025-ps3c-test-dataset-annotated.csv
        isbi2025-ps3c-eval-dataset-annotated.csv
    each with columns: image_name, label, Usage.
    """
    labels_dir = Path(labels_dir)
    path = labels_dir / f"isbi2025-ps3c-{split}-dataset-annotated.csv"
    if not path.exists():
        raise FileNotFoundError(f"Official label file not found: {path}")
    gt: dict[str, int] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lab = row["label"].strip().lower()
            if lab in CANON_IDX:
                gt[_normalize_name(row["image_name"])] = CANON_IDX[lab]
    return gt
