# Architecture Notes

## The collapse the paper addresses

The original PS3C challenge reports a striking gap between test-set and
evaluation-set F1 across every team. The driver is preprocessing-induced
distribution shift: the eval set was generated through a slightly different
pipeline than the test set (different smear scanner artifacts, different
cropping margins, different normalization). No team's training distribution
covered the eval distribution, and no team adapted at inference time.

A single number anchors the contribution:



> Recovery of the test→eval F1 drop without any retraining or eval labels.

## Why three stages, not one

A single TTA pass on an ensemble cannot be optimal because the seven teams
have radically different inductive biases. ViT models lack BatchNorm
entirely, so TENT does not apply; CNN models lack LayerNorm in the
attention sense, so the BSc-thesis method does not apply. So:

* **Stage 1** dispatches each team to the right adaptation surface.
* **Stage 2** then fuses adapted probabilities with per-sample weights,
  because not every cell type benefits equally from every team.
* **Stage 3** wraps the fused output in a conformal sieve so that bothcells
  — which no team is trained for — get deferred to a clinician with formal
  coverage guarantees.

## Family dispatch table (fixed)

| Family | Teams              | Adapter       | What it touches                            |
|--------|--------------------|---------------|--------------------------------------------|
| ViT    | JNG, YMG, NGU      | LayerNormTTA  | γ, β of every `nn.LayerNorm`               |
| CNN    | GUP, WAN           | TENT          | γ, β + recomputed running stats of `BN*d`  |
| Hybrid | DPZ, CHA           | HybridTTA     | both surfaces above, jointly               |

## Key ablations the paper must show

1. **Per-stage attribution.** Baseline → +Stage 1 → +Stage 2 → +Stage 3, on
   both test and eval. The eval-side improvement at each step is what
   reviewers will look at first.
2. **TTA surface ablation on hybrids.** For DPZ and CHA, run LayerNorm-only,
   BatchNorm-only, and combined. Tells us *which* component of a hybrid
   model is more shift-sensitive.
3. **Bothcells deferral analysis.** Show that bothcells samples are
   disproportionately routed to the deferred bin — empirical proof that the
   conformal sieve catches the failure mode the original teams ignored.
4. **Conformal coverage check.** Empirical 1−α coverage on the held-out
   test calibration split should match the nominal 95%.

## Connection to the BSc thesis

The thesis benchmarks LayerNorm TTA on PathMNIST, DermaMNIST, and
RetinaMNIST — synthetic-shift datasets where the contribution of LN-only
adaptation can be cleanly measured. This paper reuses the same mechanism
on a real clinical pipeline with a *naturally occurring* preprocessing
shift, and combines it with TENT and conformal selection to make the
result deployable.
