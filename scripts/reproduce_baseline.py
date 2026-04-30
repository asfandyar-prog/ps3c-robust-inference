"""Reproduce the seven team baselines and the gradient-boost ensemble.

Reads `configs/baseline.yaml`, runs each team model on the test and eval
splits, dumps probability arrays to `data/model_outputs/`, and writes the
per-team / ensemble F1 numbers to `results/baseline_metrics.json`.

Targets to match (from the original paper):
    JNG (best individual):  test 0.8702 / eval 0.8176
    Best ensemble (GB):     test 0.9517 / eval 0.9245
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    print("Loaded baseline config:")
    print(yaml.safe_dump(cfg, sort_keys=False))

    raise SystemExit(
        "Baseline reproduction blocked on receiving model weights / "
        "probability outputs from Prof. Harangi. See weights/README.md."
    )


if __name__ == "__main__":
    main()
