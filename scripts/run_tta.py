"""[OUT OF SCOPE] Stage 1 — architecture-aware TTA driver.

Architecture-aware TTA is out of scope for the current two-stage paper (see the
repository README, "Out of scope / Future work"). This driver and the TTA code in
`src/ps3c_robust/tta/` are retained for future work and are NOT part of the active
run_0N pipeline; nothing in the active path imports or calls this module.

Dispatches each team to the right adapter based on `TEAM_FAMILIES`, runs the
relevant ablations from `configs/tta.yaml`, and writes adapted probability
arrays to `data/model_outputs/tta/`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ps3c_robust.baseline.models import TEAM_FAMILIES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/tta.yaml"))
    args = parser.parse_args()

    with args.config.open() as f:
        cfg = yaml.safe_load(f)

    print(f"TTA dispatch table:\n{TEAM_FAMILIES}")
    print(f"Ablations to run: {[a['name'] for a in cfg['ablations']]}")

    raise SystemExit(
        "Stage 1 TTA blocked on baseline model weights. "
        "Run scripts/reproduce_baseline.py once weights arrive."
    )


if __name__ == "__main__":
    main()
