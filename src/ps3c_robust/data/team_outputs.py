"""Loader for the six PS3C team probability outputs shared by the organizers.

This reconciles the real files as received, which have several quirks that must
be handled explicitly (every one of these is confirmed from the file headers):

* JNG filenames are swapped — "Test-set.csv" contains eval_image_* rows and
  "Evaluation-set.csv" contains test_image_* rows. We assign the split by the
  image-name prefix, never by the filename.
* Four different column orderings across teams — we remap by header name.
* DPZ has no label column and its scores do not sum to one (two-stage sigmoid
  method); we renormalize over the three canonical classes.
* YMG's test file has a bothcells column; its eval file does not. We drop
  bothcells and renormalize over the kept three classes.
* WAN filenames say "validation" but the contents are test/eval (by prefix).

Canonical class order used everywhere downstream: [healthy, unhealthy, rubbish].

IMPORTANT LIMITATION: the per-team files each carry their own `label` column,
and these disagree across teams (175 cases on test, 496 on eval). They are NOT
an authoritative ground truth. Until the official APACC label file is obtained,
labels returned here are a best-effort consensus and F1 numbers computed from
them will not exactly match the published paper. See `load_labels`.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

CANON = ["healthy", "unhealthy", "rubbish"]
CANON_IDX = {c: i for i, c in enumerate(CANON)}

# Map each team's probability-column header (lowercased) to a canonical class.
HEADER_ALIASES = {
    "healthy": "healthy", "healthy_prob": "healthy", "prob_healthy": "healthy",
    "unhealthy": "unhealthy", "unhealthy_prob": "unhealthy", "prob_unhealthy": "unhealthy",
    "rubbish": "rubbish", "rubbish_prob": "rubbish", "prob_rubbish": "rubbish",
    "bothcells": "bothcells", "bothcells_prob": "bothcells",
}

TEAMS = ["YMG", "JNG", "CHA", "GUP", "DPZ", "WAN"]


def _team_file_map(data_dir: Path) -> dict[str, dict[str, Path]]:
    """Return {team: {split: path}} for the data arranged as received.

    `data_dir` should contain the unzipped organizer share plus the YMG/WAN
    folders from the follow-up archive, i.e.:

        data_dir/
        ├── wrapup_UT/wrapup_UT/...                (YMG)
        ├── jianght_challenge_materials/...        (JNG)
        ├── ChatoLina_challenge_materials/...      (CHA)
        ├── Chinmay materials/ISBI PS3C IIT KGP/.. (GUP)
        ├── Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/...  (DPZ)
        ├── huina_wang_materials/...               (WAN, original)
        ├── WAN/...                                 (WAN, follow-up)
        └── YMG/...                                 (YMG, follow-up duplicate)
    """
    d = data_dir
    return {
        "YMG": {
            "test": d / "wrapup_UT/wrapup_UT/test_phase_prob.csv",
            "eval": d / "wrapup_UT/wrapup_UT/final_eval_phase_revised_prob.csv",
        },
        "JNG": {
            # Swap: contents, not filename, decide the split.
            "test": d / "jianght_challenge_materials/Evaluation-set.csv",
            "eval": d / "jianght_challenge_materials/Test-set.csv",
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
    if n.lower().endswith(".png"):
        n = n[:-4]
    return n


def _load_team_file(path: Path) -> dict[str, np.ndarray]:
    """Read one CSV -> {image_name: prob[healthy, unhealthy, rubbish]}.

    Handles arbitrary column order, optional bothcells (dropped), and rows that
    do not sum to one (renormalized).
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Expected team file not found: {path}\n"
            "Check that the data directory matches the organizer share layout."
        )
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows = list(reader)

    col_to_class: dict[int, str] = {}
    for i, col in enumerate(header):
        key = col.strip().lower()
        if key in HEADER_ALIASES:
            col_to_class[i] = HEADER_ALIASES[key]

    out: dict[str, np.ndarray] = {}
    for row in rows:
        if not row or not row[0].strip():
            continue
        name = _normalize_name(row[0])
        vec = np.zeros(3, dtype=np.float64)
        for i, cls in col_to_class.items():
            if cls == "bothcells":
                continue
            try:
                vec[CANON_IDX[cls]] = float(row[i])
            except (ValueError, IndexError):
                vec[CANON_IDX[cls]] = 0.0
        s = vec.sum()
        if s > 0:
            vec /= s
        out[name] = vec
    return out


def load_split(data_dir: str | Path, split: str):
    """Load all six teams for one split, aligned by common image name.

    Returns:
        names:  list[str]  aligned image names (sorted)
        probs:  (N, 6, 3)  per-team probabilities in canonical class order
        labels: (N,)       consensus labels (see limitation note in module docstring)
    """
    if split not in {"test", "eval"}:
        raise ValueError(f"split must be 'test' or 'eval', got {split!r}")

    data_dir = Path(data_dir)
    files = _team_file_map(data_dir)

    per_team = {t: _load_team_file(files[t][split]) for t in TEAMS}

    common = set.intersection(*[set(d.keys()) for d in per_team.values()])
    names = sorted(common)

    probs = np.zeros((len(names), len(TEAMS), 3), dtype=np.float64)
    for ti, team in enumerate(TEAMS):
        d = per_team[team]
        for ni, name in enumerate(names):
            probs[ni, ti] = d[name]

    labels = load_labels(data_dir, split, names)
    return names, probs, labels


def load_labels(data_dir: str | Path, split: str, names: list[str]) -> np.ndarray:
    """Best-effort consensus labels from teams that provide a label column.

    NOT authoritative — see module docstring. Returns -1 for any image where no
    team supplied a usable 3-class label.
    """
    data_dir = Path(data_dir)
    files = _team_file_map(data_dir)
    name_set = set(names)
    votes: dict[str, dict[str, int]] = {n: {} for n in names}

    for team in ["YMG", "CHA", "GUP", "WAN", "JNG"]:  # DPZ has no label column
        path = files[team][split]
        with open(path, newline="") as f:
            reader = csv.reader(f)
            header = [h.strip().lower() for h in next(reader)]
            if "label" not in header:
                continue
            li = header.index("label")
            for row in reader:
                if not row:
                    continue
                nm = _normalize_name(row[0])
                if nm in name_set and len(row) > li:
                    lab = row[li].strip().lower()
                    votes[nm][lab] = votes[nm].get(lab, 0) + 1

    labels = np.full(len(names), -1, dtype=np.int64)
    for ni, name in enumerate(names):
        v3 = {k: c for k, c in votes[name].items() if k in CANON_IDX}
        if v3:
            labels[ni] = CANON_IDX[max(v3, key=v3.get)]
    return labels
