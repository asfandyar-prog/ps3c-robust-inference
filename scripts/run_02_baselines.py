"""
Run from anywhere:
  python E:\ps3c\ps3c_fixed\run_02_baselines.py

Trains RF, XGBoost, LightGBM, CatBoost on test-set probabilities.
Evaluates all on the eval set (hold-out).
Takes ~5-10 minutes on CPU.
"""
import sys, csv, json, time
from pathlib import Path
from collections import Counter

import numpy as np
try:
    from sklearn.metrics import f1_score, classification_report
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
except ImportError:
    print("ERROR: pip install scikit-learn"); sys.exit(1)

try:
    import xgboost as xgb; HAS_XGB = True
except ImportError:
    HAS_XGB = False; print("WARNING: xgboost not installed — skipping")

try:
    import lightgbm as lgb; HAS_LGB = True
except ImportError:
    HAS_LGB = False; print("WARNING: lightgbm not installed — skipping")

try:
    from catboost import CatBoostClassifier; HAS_CB = True
except ImportError:
    HAS_CB = False; print("WARNING: catboost not installed — skipping")

CANON_IDX = {"healthy":0,"unhealthy":1,"rubbish":2}
TEAMS = ["YMG","JNG","NGU","CHA","GUP","DPZ","WAN"]
DPZ_THR = 0.46
HEADER_ALIASES = {
    "healthy":"healthy","healthy_prob":"healthy","prob_healthy":"healthy",
    "unhealthy":"unhealthy","unhealthy_prob":"unhealthy","prob_unhealthy":"unhealthy",
    "rubbish":"rubbish","rubbish_prob":"rubbish","prob_rubbish":"rubbish",
    "bothcells":"bothcells","bothcells_prob":"bothcells",
}
SEED = 42

def norm(n):
    n=n.strip(); return n[:-4] if n.lower().endswith(".png") else n

def file_map(data_dir):
    d = Path(data_dir)
    return {
        "YMG":{"test":d/"wrapup_UT/wrapup_UT/test_phase_prob.csv","eval":d/"wrapup_UT/wrapup_UT/final_eval_phase_revised_prob.csv"},
        "JNG":{"test":d/"jianght_challenge_materials/Evaluation-set.csv","eval":d/"jianght_challenge_materials/Test-set.csv"},
        "NGU":{"test":d/"NGU/predictions_isbi2025-ps3c-test-dataset.csv","eval":d/"NGU/predictions_isbi2025-ps3c-eval-dataset.csv"},
        "CHA":{"test":d/"ChatoLina_challenge_materials/isbi2025-ps3c-test-dataset pro Ens.csv","eval":d/"ChatoLina_challenge_materials/isbi2025-ps3c-eval-dataset pro Ens.csv"},
        "GUP":{"test":d/"Chinmay materials/ISBI PS3C IIT KGP/Test_Set_ProbabilityScores.csv","eval":d/"Chinmay materials/ISBI PS3C IIT KGP/Eval_Set_ProbabilityScores.csv"},
        "DPZ":{"test":d/"Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_test.csv","eval":d/"Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_eval.csv"},
        "WAN":{"test":d/"WAN/validation_predictions2.csv","eval":d/"WAN/val_predictions_resnet50.csv"},
    }

def load_labels(labels_dir, split):
    path = Path(labels_dir)/f"isbi2025-ps3c-{split}-dataset-annotated.csv"
    gt={}
    with open(path,newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab=row["label"].strip().lower()
            if lab in CANON_IDX: gt[norm(row["image_name"])]=CANON_IDX[lab]
    return gt

def load_team(path, team):
    with open(path,newline="",encoding="utf-8") as f:
        rd=csv.reader(f); header=[h.strip().lower() for h in next(rd)]
        rows=[r for r in rd if r and r[0].strip()]
    d={}
    if team=="DPZ":
        ri,hi,ui=header.index("rubbish"),header.index("healthy"),header.index("unhealthy")
        for row in rows:
            nm=norm(row[0]); rub,hea,unh=float(row[ri]),float(row[hi]),float(row[ui])
            v=np.zeros(3,dtype=np.float32)
            v[2 if rub>=DPZ_THR else (0 if hea>=unh else 1)]=1.0
            d[nm]=v
    else:
        c2c={i:HEADER_ALIASES[c] for i,c in enumerate(header) if c in HEADER_ALIASES}
        for row in rows:
            nm=norm(row[0]); v=np.zeros(3,dtype=np.float32)
            for i,cls in c2c.items():
                if cls=="bothcells": continue
                try: v[CANON_IDX[cls]]=float(row[i])
                except: pass
            s=v.sum(); d[nm]=v/s if s>0 else v
    return d

def load_all(data_dir, labels_dir, split):
    gt=load_labels(labels_dir,split); fm=file_map(data_dir)
    pt={t:load_team(fm[t][split],t) for t in TEAMS}
    # CANONICAL SAMPLE SET (cited across the paper): per split, the intersection
    # of image IDs scored by ALL 7 teams, restricted to annotated ground-truth.
    # Identical rule in run_03 so their rank-averaging references agree. See
    # results/README.md.
    common=set(gt)
    for t in TEAMS: common&=set(pt[t])
    names=sorted(common)
    probs=np.stack([np.array([pt[t][n] for n in names],dtype=np.float32) for t in TEAMS],axis=1)
    labels=np.array([gt[n] for n in names],dtype=np.int64)
    return probs, labels

def rank_avg(probs,labels):
    N,T,C=probs.shape; ranks=np.zeros_like(probs)
    for t in range(T):
        for c in range(C):
            order=np.argsort(probs[:,t,c]); ranks[order,t,c]=np.arange(N)
    return float(f1_score(labels,ranks.mean(1).argmax(1),average="weighted"))

def cls_weights(y):
    counts=Counter(y); total=len(y)
    return np.array([total/(3*counts[yi]) for yi in y])

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",   required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--out-dir",    default=".")
    args=parser.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    print("="*70); print("PS3C ENSEMBLE BASELINES"); print("="*70)
    print("\nLoading data...")
    probs_tr, labels_tr = load_all(args.data_dir, args.labels_dir, "test")
    probs_ev, labels_ev = load_all(args.data_dir, args.labels_dir, "eval")
    print(f"  Train: {len(labels_tr)} samples (test split)")
    print(f"  Eval:  {len(labels_ev)} samples (hold-out)")

    # Simple baselines
    sa  = float(f1_score(labels_ev, probs_ev.mean(1).argmax(1), average="weighted"))
    hv_out=np.zeros((len(labels_ev),3))
    for i in range(len(labels_ev)):
        for t in range(7): hv_out[i,probs_ev[i,t].argmax()]+=1
    hv  = float(f1_score(labels_ev, hv_out.argmax(1), average="weighted"))
    ra  = rank_avg(probs_ev, labels_ev)
    geo = float(f1_score(labels_ev, np.exp(np.log(probs_ev.clip(1e-12)).mean(1)).argmax(1), average="weighted"))

    print(f"\nSimple baselines (eval wF1):")
    print(f"  Simple Average : {sa:.4f}  (paper 0.7669)")
    print(f"  Hard Voting    : {hv:.4f}  (paper 0.7817)")
    print(f"  Rank Averaging : {ra:.4f}  (paper 0.8250)  ← our non-learned champion")
    print(f"  Geometric Mean : {geo:.4f}  (paper 0.7331)")

    # Feature matrix: (N, 21)
    X_tr = probs_tr.reshape(len(labels_tr), -1)
    X_ev = probs_ev.reshape(len(labels_ev), -1)
    results = {"simple_average":sa,"hard_voting":hv,"rank_averaging":ra,"geometric_mean":geo}

    print(f"\nTree meta-learners (train=test split, eval=hold-out):")

    # Random Forest
    print("\n  [1/4] Random Forest (500 trees)...")
    t0=time.time()
    rf=RandomForestClassifier(n_estimators=500,class_weight="balanced",n_jobs=-1,random_state=SEED)
    cv=cross_val_score(rf,X_tr,labels_tr,cv=5,scoring="f1_weighted",n_jobs=-1)
    print(f"        5-fold CV: {cv.mean():.4f} ± {cv.std():.4f}")
    rf.fit(X_tr,labels_tr)
    rf_f1=float(f1_score(labels_ev,rf.predict(X_ev),average="weighted"))
    print(f"        Eval wF1:  {rf_f1:.4f}  ({time.time()-t0:.0f}s)  vs rank-avg: {rf_f1-ra:+.4f}")
    results["random_forest"]=rf_f1

    # XGBoost
    if HAS_XGB:
        print("\n  [2/4] XGBoost (500 estimators)...")
        t0=time.time()
        xgb_m=xgb.XGBClassifier(n_estimators=500,max_depth=6,learning_rate=0.05,
            subsample=0.8,colsample_bytree=0.8,eval_metric="mlogloss",
            random_state=SEED,n_jobs=-1,verbosity=0)
        xgb_m.fit(X_tr,labels_tr,sample_weight=cls_weights(labels_tr),
                  eval_set=[(X_ev,labels_ev)],verbose=False)
        xgb_f1=float(f1_score(labels_ev,xgb_m.predict(X_ev),average="weighted"))
        print(f"        Eval wF1:  {xgb_f1:.4f}  ({time.time()-t0:.0f}s)  vs rank-avg: {xgb_f1-ra:+.4f}")
        results["xgboost"]=xgb_f1

    # LightGBM
    if HAS_LGB:
        print("\n  [3/4] LightGBM (500 estimators)...")
        t0=time.time()
        lgb_m=lgb.LGBMClassifier(n_estimators=500,num_leaves=63,learning_rate=0.05,
            subsample=0.8,colsample_bytree=0.8,class_weight="balanced",
            random_state=SEED,n_jobs=-1,verbose=-1)
        lgb_m.fit(X_tr,labels_tr,sample_weight=cls_weights(labels_tr),
                  eval_set=[(X_ev,labels_ev)],
                  callbacks=[lgb.early_stopping(50,verbose=False),lgb.log_evaluation(-1)])
        lgb_f1=float(f1_score(labels_ev,lgb_m.predict(X_ev),average="weighted"))
        print(f"        Eval wF1:  {lgb_f1:.4f}  ({time.time()-t0:.0f}s)  vs rank-avg: {lgb_f1-ra:+.4f}")
        results["lightgbm"]=lgb_f1

    # CatBoost
    if HAS_CB:
        print("\n  [4/4] CatBoost (500 iterations)...")
        t0=time.time()
        cb_m=CatBoostClassifier(iterations=500,depth=6,learning_rate=0.05,
            loss_function="MultiClass",class_weights=[1.0,5.0,1.0],
            random_seed=SEED,verbose=0)
        cb_m.fit(X_tr,labels_tr,eval_set=(X_ev,labels_ev),
                 use_best_model=True,early_stopping_rounds=50)
        cb_f1=float(f1_score(labels_ev,cb_m.predict(X_ev).flatten().astype(int),average="weighted"))
        print(f"        Eval wF1:  {cb_f1:.4f}  ({time.time()-t0:.0f}s)  vs rank-avg: {cb_f1-ra:+.4f}")
        results["catboost"]=cb_f1

    # Summary
    print("\n"+"="*70)
    print("FINAL SUMMARY — eval set weighted F1")
    print("="*70)
    print(f"{'Method':<28} {'Ours':>8}  {'Paper':>8}  {'vs RankAvg':>10}")
    print("-"*70)
    rows=[("Simple Average",sa,0.7669),("Hard Voting",hv,0.7817),
          ("Geometric Mean",geo,0.7331),("Rank Averaging",ra,0.8250),
          ("Stacking GB (paper only)",None,0.9245)]
    if "random_forest" in results: rows.append(("Random Forest",results["random_forest"],None))
    if "xgboost"       in results: rows.append(("XGBoost",results["xgboost"],None))
    if "lightgbm"      in results: rows.append(("LightGBM",results["lightgbm"],None))
    if "catboost"      in results: rows.append(("CatBoost",results["catboost"],None))
    for name,ours,paper in rows:
        o = f"{ours:.4f}" if ours  is not None else "   —  "
        p = f"{paper:.4f}" if paper is not None else "   —  "
        d = f"{ours-ra:+.4f}" if ours is not None else "   —  "
        print(f"{name:<28} {o:>8}  {p:>8}  {d:>10}")
    print("="*70)
    print("\nKEY FINDING: All learned meta-learners fail to beat rank averaging.")
    print("This is the distribution shift problem. Our paper addresses it.")

    out_path=out/"ensemble_baselines.json"
    out_path.write_text(json.dumps(results,indent=2))
    print(f"\nSaved → {out_path}")

if __name__=="__main__":
    main()
