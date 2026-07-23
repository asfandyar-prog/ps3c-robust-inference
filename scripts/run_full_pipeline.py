"""Legacy scaffold orchestrator (baseline → ensemble → selective prediction).

Stage 1 (TTA) is out of scope, so this no longer references it. The active
two-stage pipeline is the numbered scripts run_01_verify → run_02_baselines →
run_03_weight_generator → run_04_conformal → run_05_bbse → run_06_bbse_matched,
run individually with --data-dir/--labels-dir. This orchestrator is retained for
reference only and still points at the early scaffold stubs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

STAGES = [
    ("Baseline reproduction", "scripts/reproduce_baseline.py"),
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
