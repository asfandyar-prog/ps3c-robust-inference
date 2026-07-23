"""[SUPERSEDED SCAFFOLD] Early Stage-2 driver stub.

Superseded by the active Stage-2 scripts — run_02_baselines.py,
run_03_weight_generator.py and train_adaptive_ensemble.py — which operate directly
on the raw team probability vectors. Stage 1 (TTA) is out of scope, so Stage 2 does
not depend on any adapted inputs. Retained for reference only.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/ensemble.yaml"))
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    print(yaml.safe_dump(cfg, sort_keys=False))
    raise SystemExit(
        "Superseded scaffold — use scripts/run_02_baselines.py, "
        "scripts/run_03_weight_generator.py, or scripts/train_adaptive_ensemble.py."
    )


if __name__ == "__main__":
    main()
