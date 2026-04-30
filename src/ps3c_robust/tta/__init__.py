"""Stage 1 — Architecture-aware test-time adaptation.

Three mechanisms, dispatched by model family:

* `LayerNormTTA`  — ViT models (JNG, YMG, NGU)
* `TENT`          — CNN models (GUP, WAN)
* `HybridTTA`     — Hybrid models (DPZ, CHA), applies both surfaces
"""

from ps3c_robust.tta.hybrid_tta import HybridTTA
from ps3c_robust.tta.layernorm_tta import (
    LayerNormTTA,
    mean_softmax_entropy,
    softmax_entropy,
)
from ps3c_robust.tta.tent import TENT

__all__ = [
    "TENT",
    "HybridTTA",
    "LayerNormTTA",
    "mean_softmax_entropy",
    "softmax_entropy",
]
