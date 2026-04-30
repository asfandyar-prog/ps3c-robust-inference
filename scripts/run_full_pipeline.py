"""End-to-end pipeline: baseline → TTA → ensemble → selective prediction.

Runs all four stages sequentially with their default config files. Use this
once each stage is independently working.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

STAGES = [
    ("Baseline reproduction", "scripts/reproduce_baseline.py"),
    ("Stage 1 — TTA",         "scripts/run_tta.py"),
    ("Stage 2 — Ensemble",    "scripts/train_ensemble.py"),
    ("Stage 3 — Selective",   "scripts/calibrate_conformal.py"),
]


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    for label, script in STAGES:
        print(f"\n{'=' * 70}\n{label}: {script}\n{'=' * 70}")
        result = subprocess.run([sys.executable, str(repo_root / script)], check=False)
        if result.returncode != 0:
            print(f"Stage failed: {label}")
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
