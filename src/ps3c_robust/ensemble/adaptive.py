"""Sample-adaptive attention ensemble over the seven team outputs.

Input shape:  (B, T, C)  — batch, teams, classes (T = 7, C = 4 for PS3C)
Output shape: (B, C)     — fused class probabilities
Side output:  (B, T)     — per-sample attention weights for interpretability
"""

from __future__ import annotations
import torch
from torch import Tensor, nn
class AdaptiveEnsemble(nn.Module):
    """Lightweight attention head that produces per-sample weights over teams.

    The features fed to the attention scorer for each (sample, team) pair are
    concatenations of:
        * the team's softmax probabilities (C dims)
        * confidence statistics: max-prob, entropy, top-1/top-2 margin (3 dims)

    These get projected to a hidden dim and scored against a learned global
    query — one scalar weight per team — followed by softmax across teams.
    """

    def __init__(
        self,
        num_teams: int = 7,
        num_classes: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_teams = num_teams
        self.num_classes = num_classes

        feature_dim = num_classes + 3   # probs + 3 stats
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query = nn.Parameter(torch.randn(hidden_dim) / hidden_dim**0.5)

    def forward(self, team_probs: Tensor) -> tuple[Tensor, Tensor]:
        """Fuse team probabilities for each sample.

        Args:
            team_probs: (B, T, C) softmax probabilities per team.

        Returns:
            fused:   (B, C) fused probabilities.
            weights: (B, T) per-sample attention weights over teams.
        """
        _, T, C = team_probs.shape
        if T != self.num_teams or C != self.num_classes:
            raise ValueError(
                f"Expected (*, {self.num_teams}, {self.num_classes}), got {team_probs.shape}"
            )

        stats = self._team_stats(team_probs)                # (B, T, 3)
        feats = torch.cat([team_probs, stats], dim=-1)      # (B, T, C+3)
        h = self.feature_proj(feats)                        # (B, T, hidden)

        scores = (h * self.query).sum(dim=-1)               # (B, T)
        weights = torch.softmax(scores, dim=-1)             # (B, T)

        fused = (weights.unsqueeze(-1) * team_probs).sum(dim=1)   # (B, C)
        return fused, weights

    @staticmethod
    def _team_stats(team_probs: Tensor) -> Tensor:
        """Return (max_prob, entropy, top1-top2_margin) per team."""
        max_prob = team_probs.max(dim=-1).values
        log_p = torch.log(team_probs.clamp_min(1e-12))
        entropy = -(team_probs * log_p).sum(dim=-1)
        sorted_p, _ = team_probs.sort(dim=-1, descending=True)
        margin = sorted_p[..., 0] - sorted_p[..., 1]
        return torch.stack([max_prob, entropy, margin], dim=-1)
