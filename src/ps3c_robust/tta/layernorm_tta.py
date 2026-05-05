"""LayerNorm test-time adaptation for Vision Transformers.

Ported from the Asfand Yar BSc thesis:

    Predictive Self-Supervised Vision Transformers under Test-Time
     Distribution Shifts with Lightweight TTA  (Asfand Yar, 2026)

The thesis implementation is the canonical version; this module is a
faithful port into the PS3C project structure.

Mechanism
---------
At test time, freeze the entire backbone and update only the affine
parameters (γ, β) of every `nn.LayerNorm`. The objective is entropy
minimization on the unlabelled test batch — the standard label-free TTA
loss from TENT (Wang et al., ICLR 2021), but applied to LayerNorm rather
than BatchNorm because Vision Transformers have no BN layers.

Differences from TENT
---------------------
* TENT  → BatchNorm γ/β + recomputed running statistics  → CNNs
* This  → LayerNorm γ/β only (no statistics)             → ViTs

Total adapted parameters for ViT-B/16: 2 × 768 × 12 = 18,432
(less than 0.02% of the 86M backbone parameters).

Design choices preserved from the thesis
----------------------------------------
* `copy.deepcopy(model)` — full model copy, not state-dict snapshot.
  Bulletproof when the model has nested frozen submodules (as it does
  for our JEPA-style classifiers and several PS3C team architectures).
* Optimizer is **reinstantiated** on every reset, because
  `load_state_dict` replaces parameter tensors and would leave the
  optimizer holding stale references.
* Optional `entropy_threshold`: skip adaptation on already-confident
  samples. The same threshold semantics drive the deferral decision
  in Stage 3 (selective prediction).
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# Entropy helpers — exposed so TENT and HybridTTA can share the same
# numerically stable formulation.
# ---------------------------------------------------------------------------


def softmax_entropy(logits: Tensor) -> Tensor:
    """Per-sample softmax entropy in nats.

    H(p) = -Σ_c p_c log p_c, computed via log-softmax for numerical stability.

    Args:
        logits: (B, C) unnormalised class logits.

    Returns:
        (B,) per-sample entropy.
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def mean_softmax_entropy(logits: Tensor) -> Tensor:
    """Batch mean of `softmax_entropy`. Used as the TTA loss."""
    return softmax_entropy(logits).mean()


# Backwards-compat alias for older callers in this codebase.
_softmax_entropy = softmax_entropy


# ---------------------------------------------------------------------------
# Parameter collection helpers
# ---------------------------------------------------------------------------


def collect_layernorm_params(
    model: nn.Module,
    requires_grad: bool = True,
) -> list[nn.Parameter]:
    """Collect every LayerNorm γ (weight) and β (bias) tensor.

    Args:
        model:         model to inspect.
        requires_grad: if True, also flip `requires_grad=True` on each tensor.

    Returns:
        List of LayerNorm affine parameter tensors.
    """
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                if requires_grad:
                    module.weight.requires_grad_(True)
                params.append(module.weight)
            if module.bias is not None:
                if requires_grad:
                    module.bias.requires_grad_(True)
                params.append(module.bias)
    return params


def freeze_non_layernorm_params(model: nn.Module) -> None:
    """Freeze every parameter, then unfreeze LayerNorm γ/β only."""
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.requires_grad_(True)
            if module.bias is not None:
                module.bias.requires_grad_(True)


# ---------------------------------------------------------------------------
# TTA wrapper
# ---------------------------------------------------------------------------


class LayerNormTTA(nn.Module):
    """Test-Time Adaptation wrapper for Vision Transformer models.

    Wraps a trained model and adapts its LayerNorm affine parameters by
    minimising prediction entropy on each test batch.

    Args:
        model: trained classifier with at least one `nn.LayerNorm` layer.
        lr: learning rate for LayerNorm parameter updates.
        steps: gradient steps per call (TENT default = 1).
        episodic: if True, reset weights to pre-adaptation state after
            every call. If False, accumulate adaptation across batches
            ("continual TTA"). Episodic is safer for evaluation; continual
            is more aggressive but can drift.
        entropy_threshold: if not None, only adapt on samples whose entropy
            exceeds this value. Already-confident samples are excluded from
            the adaptation loss. Set to None to adapt on all samples.

    Example:
        tta_model = LayerNormTTA(trained_classifier, lr=1e-4, steps=1)
        logits = tta_model(test_batch)   # adapts then predicts
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        steps: int = 1,
        episodic: bool = True,
        entropy_threshold: float | None = None,
    ) -> None:
        super().__init__()

        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        self.lr = lr
        self.steps = steps
        self.episodic = episodic
        self.entropy_threshold = entropy_threshold

        # Deep copy so the caller's model is never mutated.
        self.model = copy.deepcopy(model)

        # Snapshot the entire state dict for episodic reset.
        self._original_state = copy.deepcopy(self.model.state_dict())

        # Freeze everything except LayerNorm affine parameters.
        freeze_non_layernorm_params(self.model)

        # Collect trainable parameters and build the optimizer.
        self._params = collect_layernorm_params(self.model, requires_grad=True)
        if not self._params:
            raise RuntimeError(
                "No LayerNorm parameters found in model. "
                "Use TENT for BatchNorm-only models or HybridTTA for mixed models."
            )
        self.optimizer = torch.optim.Adam(self._params, lr=self.lr)

    # ------------------------------------------------------------------ public

    def reset(self) -> None:
        """Restore model weights to the pre-adaptation state.

        Called automatically before each forward pass when `episodic=True`.
        Reinstantiates the optimizer because `load_state_dict` replaces
        parameter tensors in-place — the old optimizer would otherwise be
        holding references to detached tensors.
        """
        self.model.load_state_dict(self._original_state)
        self._params = collect_layernorm_params(self.model, requires_grad=True)
        self.optimizer = torch.optim.Adam(self._params, lr=self.lr)

    @torch.enable_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Adapt on `x`, then return logits computed with the adapted weights.

        Steps:
            1. If episodic, reset to pre-adaptation weights.
            2. Run `self.steps` entropy-minimization gradient updates.
            3. Final forward pass with adapted weights, returned to caller.

        Args:
            x: (B, ...) test batch.

        Returns:
            (B, C) class logits.
        """
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            self.model.train()
            logits = self.model(x)

            if self.entropy_threshold is not None:
                batch_entropy = softmax_entropy(logits)
                mask = batch_entropy >= self.entropy_threshold
                if mask.sum() == 0:
                    # Every sample already confident — skip the update.
                    break
                loss = softmax_entropy(logits[mask]).mean()
            else:
                loss = mean_softmax_entropy(logits)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        # Final inference with adapted weights.
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

    # ---------------------------------------------------------------- diagnostics

    @property
    def num_adaptable_params(self) -> int:
        """Total number of γ/β scalars adapted at test time."""
        return sum(p.numel() for p in self._params)

    def __repr__(self) -> str:
        thr = self.entropy_threshold
        thr_str = f"{thr:.3f}" if thr is not None else "None"
        return (
            f"LayerNormTTA("
            f"steps={self.steps}, "
            f"lr={self.lr:.2e}, "
            f"episodic={self.episodic}, "
            f"entropy_threshold={thr_str}, "
            f"adaptable_params={self.num_adaptable_params:,})"
        )
