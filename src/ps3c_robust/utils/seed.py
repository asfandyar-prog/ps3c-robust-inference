"""Reproducibility helper."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA).

    Set `deterministic=True` for the final paper runs; it is slower but
    eliminates the last sources of nondeterminism in convolutions.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
