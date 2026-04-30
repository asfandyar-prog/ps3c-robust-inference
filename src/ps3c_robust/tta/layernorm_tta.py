"""LayerNorm test-time adaptation for Vision Transformers.

Adapted from the developer's BSc thesis:

    Predictive Self-Supervised Vision Transformers under Test-Time
    Distribution Shifts with Lightweight TTA

Idea: at test time, freeze the entire backbone and update only the affine
parameters (γ, β) of every LayerNorm. The objective is entropy minimization on
the unlabelled test batch — the standard label-free TTA loss.

Compared to TENT (BatchNorm), this is the right adaptation surface for
transformer-style models that have no BN layers.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn


class LayerNormTTA:
    """Wraps a ViT model with episodic LayerNorm-only entropy minimization."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        steps: int = 1,
        episodic: bool = True,
    ) -> None:
        self.model = model
        self.steps = steps
        self.episodic = episodic

        self._configure_model_for_tta()
        self._initial_state = self._snapshot_state()

        params = list(self._collect_layernorm_params())
        if not params:
            raise RuntimeError("No LayerNorm parameters found — wrong model family?")
        self.optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    # ------------------------------------------------------------------ public

    @torch.enable_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """Adapt on `x` then return the post-adaptation logits."""
        if self.episodic:
            self._restore_state()

        for _ in range(self.steps):
            logits = self.model(x)
            loss = _softmax_entropy(logits).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            return self.model(x)

    # ----------------------------------------------------------------- internal

    def _configure_model_for_tta(self) -> None:
        """Freeze everything except LayerNorm γ/β and put model in eval mode.

        We still need eval() for things like dropout to behave at inference time;
        LayerNorm has no train/eval-mode-dependent statistics, so eval() is safe.
        """
        self.model.eval()
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                if module.bias is not None:
                    module.bias.requires_grad_(True)

    def _collect_layernorm_params(self) -> Iterator[nn.Parameter]:
        for module in self.model.modules():
            if isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    yield module.weight
                if module.bias is not None:
                    yield module.bias

    def _snapshot_state(self) -> dict[str, Tensor]:
        return {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def _restore_state(self) -> None:
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._initial_state:
                    param.copy_(self._initial_state[name])


def _softmax_entropy(logits: Tensor) -> Tensor:
    """Per-sample softmax entropy."""
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)
