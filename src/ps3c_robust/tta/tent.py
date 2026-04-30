"""TENT — Test-time entropy minimization on BatchNorm parameters.

Reference: Wang et al., "Tent: Fully Test-Time Adaptation by Entropy
Minimization", ICLR 2021.

We adapt the affine parameters of every BatchNorm layer and let BN recompute
running statistics from the test batch. This is the right adaptation surface
for the CNN teams (GUP, WAN). Transformer parts of hybrid models do NOT have
BatchNorm — for those we additionally use `LayerNormTTA` (see `HybridTTA`).
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from ps3c_robust.tta.layernorm_tta import _softmax_entropy

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class TENT:
    """Wraps a CNN model with episodic BatchNorm-based entropy minimization."""

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

        params = list(self._collect_bn_params())
        if not params:
            raise RuntimeError(
                "No BatchNorm parameters found — likely a pure ViT. "
                "Use LayerNormTTA instead."
            )
        self.optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))

    @torch.enable_grad()
    def __call__(self, x: Tensor) -> Tensor:
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
        """Freeze all weights, then unfreeze BN affine params.

        Crucially, BN modules stay in `train()` mode so that they recompute
        statistics from the test batch. Everything else stays in eval.
        """
        self.model.eval()
        self.model.requires_grad_(False)
        for module in self.model.modules():
            if isinstance(module, _BN_TYPES):
                module.train()                 # use batch statistics
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                if module.bias is not None:
                    module.bias.requires_grad_(True)

    def _collect_bn_params(self) -> Iterator[nn.Parameter]:
        for module in self.model.modules():
            if isinstance(module, _BN_TYPES):
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
