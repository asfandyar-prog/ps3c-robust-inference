"""Stage 2 — train the sample-adaptive ensemble.

Reads adapted probability arrays produced by Stage 1, fits the
`AdaptiveEnsemble` attention head, and saves both the trained weights and
the per-sample team-attention weights for later analysis.
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
    raise SystemExit("Stage 2 blocked on Stage 1 outputs.")


if __name__ == "__main__":
    main()
