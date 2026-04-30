"""APACC dataset and team-probability loaders.

Two responsibilities:

* `APACCDataset`            — image dataset for any team that needs to run
                              inference end-to-end (Stage 1 with adaptation).
* `load_team_probabilities` — load the per-team softmax outputs supplied by
                              Prof. Harangi for Stages 2 and 3.

Until weights/probabilities arrive both functions raise informative errors
that point at the expected layout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

CLASS_NAMES = ["healthy", "unhealthy", "rubbish", "bothcells"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

Split = Literal["train", "test", "eval"]


class APACCDataset:
    """Minimal stub — flesh out once we know the on-disk layout from Prof. Harangi.

    Expected structure:
        data/raw/apacc/<split>/<class_name>/<image_file>
    """

    def __init__(
        self,
        root: str | Path,
        split: Split,
        image_size: int = 224,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.split_dir = self.root / split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Expected split directory {self.split_dir}. "
                "See data/README.md for the expected APACC layout."
            )

        self.samples = self._index_samples()

    def _index_samples(self) -> list[tuple[Path, int]]:
        items: list[tuple[Path, int]] = []
        for class_name, class_idx in CLASS_TO_IDX.items():
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                continue
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                    items.append((img_path, class_idx))
        return items

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        # Implementation pending — depends on the preprocessing each team used.
        raise NotImplementedError(
            "APACCDataset.__getitem__ to be implemented after we know the "
            "per-team preprocessing pipeline that produced the probability files."
        )


def load_team_probabilities(
    probs_dir: str | Path,
    teams: list[str],
    split: Literal["test", "eval"],
) -> np.ndarray:
    """Stack per-team probability files into a (N, T, C) array.

    Args:
        probs_dir: directory containing `<team>_<split>_probs.npy` files.
        teams:     ordered list of team identifiers; output column order matches.
        split:     'test' or 'eval'.

    Returns:
        (N, T, C) float32 array of softmax probabilities.
    """
    probs_dir = Path(probs_dir)
    arrays = []
    for team in teams:
        path = probs_dir / f"{team}_{split}_probs.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing probability file: {path}. "
                "These come from Prof. Harangi — see data/README.md."
            )
        arrays.append(np.load(path).astype(np.float32))

    # All teams must agree on row count and class count.
    shapes = {a.shape for a in arrays}
    if len({s[0] for s in shapes}) != 1:
        raise ValueError(f"Inconsistent N across teams: {shapes}")
    if len({s[1] for s in shapes}) != 1:
        raise ValueError(f"Inconsistent num_classes across teams: {shapes}")

    return np.stack(arrays, axis=1)   # (N, T, C)
