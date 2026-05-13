"""Hybrid TTA — apply LayerNorm and BatchNorm adaptation simultaneously.

Used for DPZ (SwinV2 + ConvNeXt + SE-ResNeXt) and CHA (CNN + Swin Transformer).

The interesting research question this enables — and the key ablation for the
paper — is which surface contributes more to the eval-set recovery in hybrid
architectures. The config exposes flags to ablate each independently.
"""








from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor, nn

from ps3c_robust.tta.layernorm_tta import _softmax_entropy

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)


class HybridTTA:
    """Apply LayerNorm and/or BatchNorm entropy minimization to one model."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        steps: int = 1,
        episodic: bool = True,
        adapt_layernorm: bool = True,
        adapt_batchnorm: bool = True,
    ) -> None:
        if not (adapt_layernorm or adapt_batchnorm):
            raise ValueError(
                "HybridTTA needs at least one of {layernorm, batchnorm} enabled."
            )

        self.model = model
        self.steps = steps
        self.episodic = episodic
        self.adapt_layernorm = adapt_layernorm
        self.adapt_batchnorm = adapt_batchnorm

        self._configure_model_for_tta()
        self._initial_state = self._snapshot_state()

        params = list(self._collect_params())
        if not params:
            raise RuntimeError(
                "No adaptable parameters found. "
                "Confirm the model actually has LN/BN layers."
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
        self.model.eval()
        self.model.requires_grad_(False)

        for module in self.model.modules():
            if self.adapt_layernorm and isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                if module.bias is not None:
                    module.bias.requires_grad_(True)
            elif self.adapt_batchnorm and isinstance(module, _BN_TYPES):
                module.train()
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None
                if module.weight is not None:
                    module.weight.requires_grad_(True)
                if module.bias is not None:
                    module.bias.requires_grad_(True)

    def _collect_params(self) -> Iterator[nn.Parameter]:
        for module in self.model.modules():
            if self.adapt_layernorm and isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    yield module.weight
                if module.bias is not None:
                    yield module.bias
            elif self.adapt_batchnorm and isinstance(module, _BN_TYPES):
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
