"""Stage 3 — calibrate the conformal predictor and run selective evaluation.

Splits the test set into calibration / hold-out, fits the conformal predictor,
and reports coverage, deferral rate, accepted-set F1, and bothcells deferral
rate on the eval split.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/selective.yaml"))
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    print(yaml.safe_dump(cfg, sort_keys=False))
    raise SystemExit("Stage 3 blocked on Stage 2 ensemble probabilities.")


if __name__ == "__main__":
    main()
