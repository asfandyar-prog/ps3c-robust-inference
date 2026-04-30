"""Loaders for the seven PS3C team models.

Each loader returns an `nn.Module` in eval mode on the requested device. The
actual implementations are blocked on receiving the weights and the
team-specific architecture details from Prof. Harangi.

The loaders are deliberately thin — the heavy lifting (foundation-model
backbones, LoRA adapters, ensemble heads) lives inside the team subpackages
once the weights arrive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn

# Family taxonomy drives Stage 1 (which TTA mechanism applies).
ModelFamily = Literal["vit", "cnn", "hybrid"]

TEAM_FAMILIES: dict[str, ModelFamily] = {
    "jng": "vit",       # UNI / H-optimus / Gigapath foundation models + LoRA
    "ymg": "vit",       # MaxViT
    "ngu": "vit",       # EVA-02
    "gup": "cnn",       # ResNet ensemble
    "wan": "cnn",       # ResNet50 + LIME
    "dpz": "hybrid",    # SwinV2 + ConvNeXt + SE-ResNeXt
    "cha": "hybrid",    # CNN + Swin
}


def load_team_model(
    team: str,
    weights_dir: str | Path,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Load the model for `team` and move it to `device` in eval mode."""
    if team not in TEAM_FAMILIES:
        raise ValueError(f"Unknown team {team!r}. Known: {sorted(TEAM_FAMILIES)}")

    weights_dir = Path(weights_dir)
    if not weights_dir.exists():
        raise FileNotFoundError(
            f"Weights directory {weights_dir} not found. "
            "See weights/README.md for the expected layout."
        )

    loader = _LOADERS[team]
    model = loader(weights_dir)
    return model.to(device).eval()


# ---------------------------------------------------------------------------
# Per-team loaders — stubs to be filled in when weights arrive.
# ---------------------------------------------------------------------------


def _load_jng(weights_dir: Path) -> nn.Module:
    """JNG: foundation-model ensemble (UNI + H-optimus + Gigapath) with LoRA."""
    raise NotImplementedError("JNG loader pending weights from Prof. Harangi.")


def _load_ymg(weights_dir: Path) -> nn.Module:
    """YMG: MaxViT hybrid architecture."""
    raise NotImplementedError("YMG loader pending weights.")


def _load_ngu(weights_dir: Path) -> nn.Module:
    """NGU: EVA-02 Vision Transformer."""
    raise NotImplementedError("NGU loader pending weights.")


def _load_gup(weights_dir: Path) -> nn.Module:
    """GUP: ResNet variants ensemble."""
    raise NotImplementedError("GUP loader pending weights.")


def _load_wan(weights_dir: Path) -> nn.Module:
    """WAN: ResNet50 with LIME interpretability (LIME not used at inference)."""
    raise NotImplementedError("WAN loader pending weights.")


def _load_dpz(weights_dir: Path) -> nn.Module:
    """DPZ: SwinV2 + ConvNeXt + SE-ResNeXt hybrid ensemble."""
    raise NotImplementedError("DPZ loader pending weights.")


def _load_cha(weights_dir: Path) -> nn.Module:
    """CHA: CNN + Swin Transformer hybrid."""
    raise NotImplementedError("CHA loader pending weights.")


_LOADERS = {
    "jng": _load_jng,
    "ymg": _load_ymg,
    "ngu": _load_ngu,
    "gup": _load_gup,
    "wan": _load_wan,
    "dpz": _load_dpz,
    "cha": _load_cha,
}
