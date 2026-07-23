"""Train and evaluate the sample-adaptive ensemble.

The adaptive ensemble is a lightweight attention head that produces
per-sample weights over the seven team probability outputs. It replaces
the static gradient-boost stacking from the original PS3C paper.

Architecture
------------
Input:  (B, T, C) — batch of T=7 team probability vectors, C=3 classes
For each (sample, team), we compute:
    features = [probs(C), max_prob(1), entropy(1), margin(1)] = C+3 dims
These are projected to a hidden dim, scored against a learned query,
and softmax'd across teams to produce per-sample weights.
Output: (B, C) fused probabilities  +  (B, T) interpretability weights

Training setup
--------------
- Train on 80% of the test split (14,527 images)
- Validate on 20% of the test split (3,632 images) for early stopping
- Evaluate on the full eval split (29,117 images)
- Loss: cross-entropy
- Optimizer: Adam, lr=1e-3
- Early stopping: patience=5 on val weighted-F1

This matches the paper's setup (meta-learners trained on test, evaluated
on eval) while using a held-out val split to avoid overfitting.

Usage
-----
    python scripts/train_adaptive_ensemble.py \
        --data-dir   E:/ps3c/ps3c-team-data \
        --labels-dir E:/ps3c/ps3c-labels \
        --out-dir    E:/ps3c/ps3c-results

Outputs
-------
    <out-dir>/adaptive_ensemble.pt          trained model weights
    <out-dir>/adaptive_ensemble_results.txt comparison table
    <out-dir>/attention_weights_eval.npy    (N_eval, 7) per-sample weights
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split

# ── Canonical loader constants — identical to run_03 / run_02 / the baseline
# table. DPZ is hardened via the cascade rule (thr=0.46) and the sample set is
# the 7-way intersection ∩ annotated labels (test 18,159 / eval 29,117). Only
# this data-loading path differs from the earlier, superseded version of this
# script; the model architecture and hyperparameters are unchanged.
CANON_IDX = {"healthy": 0, "unhealthy": 1, "rubbish": 2}
TEAMS = ["YMG", "JNG", "NGU", "CHA", "GUP", "DPZ", "WAN"]
DPZ_THR = 0.46
HEADER_ALIASES = {
    "healthy": "healthy", "healthy_prob": "healthy", "prob_healthy": "healthy",
    "unhealthy": "unhealthy", "unhealthy_prob": "unhealthy", "prob_unhealthy": "unhealthy",
    "rubbish": "rubbish", "rubbish_prob": "rubbish", "prob_rubbish": "rubbish",
    "bothcells": "bothcells", "bothcells_prob": "bothcells",
}

# ──────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────────────────────────────────────

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading  (returns full probability vectors, not just argmax)
# ──────────────────────────────────────────────────────────────────────────────

def _norm(n: str) -> str:
    n = n.strip()
    return n[:-4] if n.lower().endswith(".png") else n


def _file_map(data_dir: Path) -> dict:
    d = Path(data_dir)
    return {
        "YMG": {"test": d/"wrapup_UT/wrapup_UT/test_phase_prob.csv", "eval": d/"wrapup_UT/wrapup_UT/final_eval_phase_revised_prob.csv"},
        "JNG": {"test": d/"jianght_challenge_materials/Evaluation-set.csv", "eval": d/"jianght_challenge_materials/Test-set.csv"},
        "NGU": {"test": d/"NGU/predictions_isbi2025-ps3c-test-dataset.csv", "eval": d/"NGU/predictions_isbi2025-ps3c-eval-dataset.csv"},
        "CHA": {"test": d/"ChatoLina_challenge_materials/isbi2025-ps3c-test-dataset pro Ens.csv", "eval": d/"ChatoLina_challenge_materials/isbi2025-ps3c-eval-dataset pro Ens.csv"},
        "GUP": {"test": d/"Chinmay materials/ISBI PS3C IIT KGP/Test_Set_ProbabilityScores.csv", "eval": d/"Chinmay materials/ISBI PS3C IIT KGP/Eval_Set_ProbabilityScores.csv"},
        "DPZ": {"test": d/"Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_test.csv", "eval": d/"Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_eval.csv"},
        "WAN": {"test": d/"WAN/validation_predictions2.csv", "eval": d/"WAN/val_predictions_resnet50.csv"},
    }


def _load_labels(labels_dir: Path, split: str) -> dict:
    path = Path(labels_dir)/f"isbi2025-ps3c-{split}-dataset-annotated.csv"
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = row["label"].strip().lower()
            if lab in CANON_IDX:
                gt[_norm(row["image_name"])] = CANON_IDX[lab]
    return gt


def _load_team(path: Path, team: str) -> dict:
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = [h.strip().lower() for h in next(rd)]
        rows = [r for r in rd if r and r[0].strip()]
    d: dict[str, np.ndarray] = {}
    if team == "DPZ":
        ri, hi, ui = header.index("rubbish"), header.index("healthy"), header.index("unhealthy")
        for row in rows:
            nm = _norm(row[0]); rub, hea, unh = float(row[ri]), float(row[hi]), float(row[ui])
            v = np.zeros(3, dtype=np.float32)
            v[2 if rub >= DPZ_THR else (0 if hea >= unh else 1)] = 1.0
            d[nm] = v
    else:
        c2c = {i: HEADER_ALIASES[c] for i, c in enumerate(header) if c in HEADER_ALIASES}
        for row in rows:
            nm = _norm(row[0]); v = np.zeros(3, dtype=np.float32)
            for i, cls in c2c.items():
                if cls == "bothcells":
                    continue
                try:
                    v[CANON_IDX[cls]] = float(row[i])
                except (ValueError, IndexError):
                    pass
            s = v.sum(); d[nm] = v / s if s > 0 else v
    return d


def load_probs_and_labels(
    data_dir: Path,
    labels_dir: Path,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Canonical loader — identical rule to run_03 / run_02 / the baseline table.

    Sample set = 7-way intersection of image IDs ∩ annotated labels; DPZ hardened
    via the cascade rule (thr=0.46). Returns (probs (N,T,C) float32, labels (N,) int64).
    """
    gt = _load_labels(labels_dir, split)
    fm = _file_map(data_dir)
    per_team = {t: _load_team(fm[t][split], t) for t in TEAMS}
    common = set(gt)
    for t in TEAMS:
        common &= set(per_team[t])
    names = sorted(common)
    probs = np.stack(
        [np.array([per_team[t][n] for n in names], dtype=np.float32) for t in TEAMS],
        axis=1,
    )  # (N, T, C)
    labels = np.array([gt[n] for n in names], dtype=np.int64)
    return probs, labels


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveEnsemble(nn.Module):
    """Lightweight attention head over T team probability vectors.

    For each (sample, team) pair the input features are:
        - team's softmax probabilities     (C dims)
        - max probability (confidence)     (1 dim)
        - softmax entropy (uncertainty)    (1 dim)
        - top-1 minus top-2 margin         (1 dim)
    These are projected to hidden_dim, scored against a learned global
    query vector, and softmax'd across teams to give per-sample weights.
    The weighted sum of team probabilities is the fused output.
    """

    def __init__(
        self,
        num_teams: int = 7,
        num_classes: int = 3,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_teams = num_teams
        self.num_classes = num_classes
        feature_dim = num_classes + 3   # probs + 3 statistics

        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query = nn.Parameter(
            torch.randn(hidden_dim) / hidden_dim ** 0.5
        )

    def forward(
        self, team_probs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            team_probs: (B, T, C) probability vectors per team.
        Returns:
            fused:   (B, C) fused class probabilities.
            weights: (B, T) per-sample attention weights (sum to 1 across T).
        """
        B, T, C = team_probs.shape

        # Per-team statistics.
        max_prob = team_probs.max(dim=-1).values                        # (B, T)
        log_p = torch.log(team_probs.clamp(min=1e-12))
        entropy = -(team_probs * log_p).sum(dim=-1)                     # (B, T)
        sorted_p = team_probs.sort(dim=-1, descending=True).values
        margin = sorted_p[..., 0] - sorted_p[..., 1]                   # (B, T)

        stats = torch.stack([max_prob, entropy, margin], dim=-1)        # (B, T, 3)
        feats = torch.cat([team_probs, stats], dim=-1)                  # (B, T, C+3)

        h = self.feature_proj(feats)                                    # (B, T, hidden)
        scores = (h * self.query).sum(dim=-1)                           # (B, T)
        weights = torch.softmax(scores, dim=-1)                         # (B, T)

        fused = (weights.unsqueeze(-1) * team_probs).sum(dim=1)         # (B, C)
        return fused, weights


# ──────────────────────────────────────────────────────────────────────────────
# Simple ensemble baselines (for comparison table)
# ──────────────────────────────────────────────────────────────────────────────

def simple_average_f1(probs: np.ndarray, labels: np.ndarray) -> float:
    fused = probs.mean(axis=1)
    return float(f1_score(labels, fused.argmax(axis=1), average="weighted"))


def rank_average_f1(probs: np.ndarray, labels: np.ndarray) -> float:
    N, T, C = probs.shape
    ranks = np.zeros_like(probs)
    for t in range(T):
        for c in range(C):
            order = np.argsort(probs[:, t, c])
            ranks[order, t, c] = np.arange(N)
    fused = ranks.mean(axis=1)
    return float(f1_score(labels, fused.argmax(axis=1), average="weighted"))


def hard_voting_f1(probs: np.ndarray, labels: np.ndarray) -> float:
    votes = probs.argmax(axis=2)
    N, T = votes.shape
    out = np.zeros((N, 3))
    for i in range(N):
        for t in range(T):
            out[i, votes[i, t]] += 1
    return float(f1_score(labels, out.argmax(axis=1), average="weighted"))


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    model: AdaptiveEnsemble,
    probs_train: np.ndarray,
    labels_train: np.ndarray,
    probs_val: np.ndarray,
    labels_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 5,
    device: torch.device = torch.device("cpu"),
) -> list[dict]:
    """Train the adaptive ensemble, return per-epoch history."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    X_tr = torch.from_numpy(probs_train).to(device)
    y_tr = torch.from_numpy(labels_train).to(device)
    X_val = torch.from_numpy(probs_val).to(device)

    best_val_f1 = -1.0
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        # Shuffle
        perm = torch.randperm(len(y_tr), device=device)
        X_tr, y_tr = X_tr[perm], y_tr[perm]

        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(y_tr), batch_size):
            xb = X_tr[start: start + batch_size]
            yb = y_tr[start: start + batch_size]
            fused, _ = model(xb)
            loss = criterion(fused, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            fused_val, _ = model(X_val)
        preds_val = fused_val.argmax(dim=1).cpu().numpy()
        val_f1 = float(f1_score(labels_val, preds_val, average="weighted"))

        history.append({
            "epoch": epoch,
            "loss": epoch_loss / n_batches,
            "val_wf1": val_f1,
        })
        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"loss={epoch_loss / n_batches:.4f}  "
            f"val_wF1={val_f1:.4f}",
            end=""
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            print("  *")
        else:
            no_improve += 1
            print(f"  (no improve {no_improve}/{patience})")
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",   type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--out-dir",    type=Path, default=Path("results"))
    parser.add_argument("--epochs",     type=int,  default=50)
    parser.add_argument("--batch-size", type=int,  default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,  default=5)
    parser.add_argument("--hidden-dim", type=int,  default=64)
    parser.add_argument("--val-frac",   type=float, default=0.2)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Teams:  {TEAMS}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading test split (used for meta-learner training)...")
    probs_test, labels_test = load_probs_and_labels(
        args.data_dir, args.labels_dir, "test"
    )
    print(f"  {len(labels_test)} images, {probs_test.shape[1]} teams, "
          f"{probs_test.shape[2]} classes\n")

    print("Loading eval split (held-out evaluation)...")
    probs_eval, labels_eval = load_probs_and_labels(
        args.data_dir, args.labels_dir, "eval"
    )
    print(f"  {len(labels_eval)} images\n")

    # ── Train / val split from test set ───────────────────────────────────────
    idx = np.arange(len(labels_test))
    idx_train, idx_val = train_test_split(
        idx, test_size=args.val_frac, random_state=SEED, stratify=labels_test
    )
    probs_tr  = probs_test[idx_train]
    labels_tr = labels_test[idx_train]
    probs_val = probs_test[idx_val]
    labels_val = labels_test[idx_val]
    print(f"Meta-learner train: {len(labels_tr)}  |  val: {len(labels_val)}\n")

    # ── Simple baselines on eval (for comparison) ─────────────────────────────
    sa_f1  = simple_average_f1(probs_eval, labels_eval)
    ra_f1  = rank_average_f1(probs_eval, labels_eval)
    hv_f1  = hard_voting_f1(probs_eval, labels_eval)
    print(f"Baseline eval weighted-F1 (for comparison):")
    print(f"  Simple Average : {sa_f1:.4f}  (paper 0.7669)")
    print(f"  Hard Voting    : {hv_f1:.4f}  (paper 0.7817)")
    print(f"  Rank Averaging : {ra_f1:.4f}  (paper 0.8250)")
    print(f"  Stacking GB    : (paper 0.9245 — trained on more data)")
    print()

    # ── Train adaptive ensemble ───────────────────────────────────────────────
    model = AdaptiveEnsemble(
        num_teams=len(TEAMS),
        num_classes=probs_test.shape[2],
        hidden_dim=args.hidden_dim,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"AdaptiveEnsemble: {n_params:,} trainable parameters\n")
    print("Training...")
    t0 = time.time()
    history = train(
        model,
        probs_tr, labels_tr,
        probs_val, labels_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        device=device,
    )
    print(f"\nTraining finished in {time.time() - t0:.1f}s")

    # ── Evaluate on eval set ──────────────────────────────────────────────────
    model.eval()
    X_ev = torch.from_numpy(probs_eval).to(device)
    with torch.no_grad():
        fused_ev, attn_ev = model(X_ev)
    preds_ev = fused_ev.argmax(dim=1).cpu().numpy()
    attn_np  = attn_ev.cpu().numpy()

    adaptive_f1 = float(f1_score(labels_eval, preds_ev, average="weighted"))
    _rec = recall_score(labels_eval, preds_ev, average=None, labels=[0, 1, 2])
    per_class_recall = {n: float(_rec[k]) for k, n in enumerate(["healthy", "unhealthy", "rubbish"])}
    best_val_f1 = max((h["val_wf1"] for h in history), default=float("nan"))

    # ── Results table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS — eval set weighted-F1")
    print("=" * 60)
    rows = [
        ("Simple Average",        sa_f1,        0.7669),
        ("Hard Voting",           hv_f1,        0.7817),
        ("Rank Averaging",        ra_f1,        0.8250),
        ("Stacking GB (paper)",   float("nan"), 0.9245),
        ("AdaptiveEnsemble",      adaptive_f1,  float("nan")),
    ]
    for name, ours, paper in rows:
        ours_str  = f"{ours:.4f}" if not np.isnan(ours)  else "  n/a "
        paper_str = f"{paper:.4f}" if not np.isnan(paper) else "  n/a "
        print(f"  {name:<28} ours={ours_str}  paper={paper_str}")
    print("=" * 60)

    # ── Save outputs ──────────────────────────────────────────────────────────
    torch.save(model.state_dict(), args.out_dir / "adaptive_ensemble.pt")
    np.save(args.out_dir / "attention_weights_eval.npy", attn_np)
    print(f"\nSaved model  -> {args.out_dir}/adaptive_ensemble.pt")
    print(f"Saved attn   -> {args.out_dir}/attention_weights_eval.npy")

    # Per-class recall (esp. unhealthy — where the shift hits hardest)
    print("\nAdaptiveEnsemble per-class recall (eval):")
    for n in ["healthy", "unhealthy", "rubbish"]:
        print(f"  {n:<10} {per_class_recall[n]:.4f}")

    # Summary + machine-readable results (canonical loader = same as run_03)
    summary = (
        f"AdaptiveEnsemble (canonical loader) eval wF1: {adaptive_f1:.4f}  |  "
        f"unhealthy recall: {per_class_recall['unhealthy']:.4f}  |  "
        f"Rank Averaging (canonical): {ra_f1:.4f}  |  "
        f"n_test={len(labels_test)} n_eval={len(labels_eval)}"
    )
    (args.out_dir / "adaptive_ensemble_results.txt").write_text(summary)
    payload = {
        "eval_wf1": round(adaptive_f1, 4),
        "per_class_recall": {k: round(v, 4) for k, v in per_class_recall.items()},
        "best_val_wf1": round(best_val_f1, 4),
        "rank_averaging_canonical": round(ra_f1, 4),
        "simple_average": round(sa_f1, 4),
        "hard_voting": round(hv_f1, 4),
        "canonical_n": {"test": int(len(labels_test)), "eval": int(len(labels_eval))},
        "loader": "canonical (run_03: 7-way intersection, DPZ hardened cascade thr=0.46)",
        "note": "Supersedes the earlier non-canonical 0.7479 "
                "(ps3c-results/adaptive_ensemble_results.txt), which used the "
                "ps3c_robust.data.team_outputs loader (soft DPZ, rank-avg ref 0.8199). "
                "Model architecture and hyperparameters unchanged; only the data-loading path differs.",
    }
    (args.out_dir / "adaptive_ensemble_canonical.json").write_text(json.dumps(payload, indent=2))
    print(f"\n{summary}")

    # Per-team attention weight analysis (which teams get trusted the most)
    print("\nMean attention weight per team on eval set:")
    mean_attn = attn_np.mean(axis=0)
    for t, w in sorted(zip(TEAMS, mean_attn), key=lambda x: -x[1]):
        print(f"  {t}: {w:.4f}")


if __name__ == "__main__":
    main()
