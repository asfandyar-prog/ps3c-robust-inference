"""
Run from anywhere:
  python E:\ps3c\ps3c_fixed\run_01_verify.py

Verifies all 7 team outputs exactly match paper Table 2.
Expected output: 14 rows all showing checkmark.
"""
import sys, csv
from pathlib import Path

# ── embedded data loader (no imports needed) ──────────────────────────────────
CANON_IDX = {"healthy": 0, "unhealthy": 1, "rubbish": 2}
TEAMS     = ["YMG", "JNG", "NGU", "CHA", "GUP", "DPZ", "WAN"]
DPZ_THR   = 0.46
HEADER_ALIASES = {
    "healthy": "healthy", "healthy_prob": "healthy", "prob_healthy": "healthy",
    "unhealthy": "unhealthy", "unhealthy_prob": "unhealthy", "prob_unhealthy": "unhealthy",
    "rubbish": "rubbish", "rubbish_prob": "rubbish", "prob_rubbish": "rubbish",
    "bothcells": "bothcells", "bothcells_prob": "bothcells",
}
PAPER = {
    "YMG": {"test_acc":0.8686,"test_f1":0.8680,"eval_acc":0.7996,"eval_f1":0.7858},
    "JNG": {"test_acc":0.8700,"test_f1":0.8702,"eval_acc":0.8229,"eval_f1":0.8176},
    "NGU": {"test_acc":0.8499,"test_f1":0.8523,"eval_acc":0.7581,"eval_f1":0.7136},
    "CHA": {"test_acc":0.8165,"test_f1":0.8319,"eval_acc":0.7953,"eval_f1":0.7944},
    "GUP": {"test_acc":0.8604,"test_f1":0.8604,"eval_acc":0.7713,"eval_f1":0.7211},
    "DPZ": {"test_acc":0.8627,"test_f1":0.8622,"eval_acc":0.7687,"eval_f1":0.7092},
    "WAN": {"test_acc":0.7901,"test_f1":0.8058,"eval_acc":0.7723,"eval_f1":0.7615},
}

import numpy as np
try:
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    print("ERROR: scikit-learn not installed. Run: pip install scikit-learn")
    sys.exit(1)

def norm(n):
    n = n.strip()
    return n[:-4] if n.lower().endswith(".png") else n

def files(data_dir):
    d = Path(data_dir)
    return {
        "YMG":{"test":d/"wrapup_UT/wrapup_UT/test_phase_prob.csv",
               "eval":d/"wrapup_UT/wrapup_UT/final_eval_phase_revised_prob.csv"},
        "JNG":{"test":d/"jianght_challenge_materials/Evaluation-set.csv",
               "eval":d/"jianght_challenge_materials/Test-set.csv"},
        "NGU":{"test":d/"NGU/predictions_isbi2025-ps3c-test-dataset.csv",
               "eval":d/"NGU/predictions_isbi2025-ps3c-eval-dataset.csv"},
        "CHA":{"test":d/"ChatoLina_challenge_materials/isbi2025-ps3c-test-dataset pro Ens.csv",
               "eval":d/"ChatoLina_challenge_materials/isbi2025-ps3c-eval-dataset pro Ens.csv"},
        "GUP":{"test":d/"Chinmay materials/ISBI PS3C IIT KGP/Test_Set_ProbabilityScores.csv",
               "eval":d/"Chinmay materials/ISBI PS3C IIT KGP/Eval_Set_ProbabilityScores.csv"},
        "DPZ":{"test":d/"Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_test.csv",
               "eval":d/"Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_eval.csv"},
        "WAN":{"test":d/"WAN/validation_predictions2.csv",
               "eval":d/"WAN/val_predictions_resnet50.csv"},
    }

def load_labels(labels_dir, split):
    path = Path(labels_dir) / f"isbi2025-ps3c-{split}-dataset-annotated.csv"
    gt = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = row["label"].strip().lower()
            if lab in CANON_IDX:
                gt[norm(row["image_name"])] = CANON_IDX[lab]
    return gt

def load_team(path, team):
    with open(path, newline="", encoding="utf-8") as f:
        rd = csv.reader(f)
        header = [h.strip().lower() for h in next(rd)]
        rows = [r for r in rd if r and r[0].strip()]
    d = {}
    if team == "DPZ":
        ri, hi, ui = header.index("rubbish"), header.index("healthy"), header.index("unhealthy")
        for row in rows:
            nm = norm(row[0])
            rub, hea, unh = float(row[ri]), float(row[hi]), float(row[ui])
            if rub >= DPZ_THR:
                pred = 2
            else:
                pred = 0 if hea >= unh else 1
            v = np.zeros(3, dtype=np.float32); v[pred] = 1.0
            d[nm] = v
    else:
        c2c = {i: HEADER_ALIASES[c] for i, c in enumerate(header) if c in HEADER_ALIASES}
        for row in rows:
            nm = norm(row[0]); v = np.zeros(3, dtype=np.float32)
            for i, cls in c2c.items():
                if cls == "bothcells": continue
                try: v[CANON_IDX[cls]] = float(row[i])
                except: pass
            s = v.sum(); d[nm] = v/s if s > 0 else v
    return d

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   required=True)
    parser.add_argument("--labels-dir", required=True)
    args = parser.parse_args()

    fm = files(args.data_dir)
    print("=" * 76)
    print("PS3C BASELINE REPRODUCTION  — verifying all 7 teams vs paper Table 2")
    print("=" * 76)
    print(f"{'Team':<6} {'Split':<5} {'Acc':>8} {'Paper':>8}  {'wF1':>8} {'Paper':>8}  {'Match'}")
    print("-" * 76)

    all_ok = True
    for team in TEAMS:
        for split in ["test","eval"]:
            gt     = load_labels(args.labels_dir, split)
            probs  = load_team(fm[team][split], team)
            common = sorted(set(gt) & set(probs))
            y_true = np.array([gt[n]             for n in common])
            y_pred = np.array([probs[n].argmax() for n in common])
            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, average="weighted")
            pa  = PAPER[team][f"{split}_acc"]
            pf  = PAPER[team][f"{split}_f1"]
            ok  = abs(acc-pa)<5e-4 and abs(f1-pf)<5e-4
            if not ok: all_ok = False
            mark = "  ✓" if ok else "  ✗ MISMATCH"
            print(f"{team:<6} {split:<5} {acc:>8.4f} {pa:>8.4f}  {f1:>8.4f} {pf:>8.4f}  {mark}  n={len(common)}")
        print()

    print("=" * 76)
    if all_ok:
        print("ALL 7 TEAMS REPRODUCED EXACTLY ✓  —  ready to proceed to Script 2")
    else:
        print("WARNING: mismatches found — check data-dir paths")
    print("=" * 76)

if __name__ == "__main__":
    main()
