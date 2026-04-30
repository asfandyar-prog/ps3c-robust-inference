"""Stage 2 — Sample-adaptive ensemble.

Replaces the fixed gradient-boost weights of the original PS3C paper with a
lightweight attention head that chooses *per-sample* weights over the seven
team probability outputs.
"""

from ps3c_robust.ensemble.adaptive import AdaptiveEnsemble

__all__ = ["AdaptiveEnsemble"]
