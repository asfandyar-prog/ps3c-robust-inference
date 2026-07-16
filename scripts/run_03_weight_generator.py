"""
Run from anywhere:
  python E:\ps3c\ps3c_fixed\run_03_weight_generator.py

Harangi's Weight Generator architecture — cluster-based variant.
Trains on test split, evaluates on eval split (hold-out).

Recommended: run with --epochs 100 on GPU machine.
Quick test: use --epochs 5 --patience 5
"""
import sys, csv, json, time, argparse
from pathlib import Path
from collections import Counter

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("ERROR: pip install torch"); sys.exit(1)
try:
    from sklearn.metrics import f1_score, classification_report
    from sklearn.model_selection import train_test_split
except ImportError:
    print("ERROR: pip install scikit-learn"); sys.exit(1)

CANON_IDX={"healthy":0,"unhealthy":1,"rubbish":2}
TEAMS=["YMG","JNG","NGU","CHA","GUP","DPZ","WAN"]
DPZ_THR=0.46
HEADER_ALIASES={
    "healthy":"healthy","healthy_prob":"healthy","prob_healthy":"healthy",
    "unhealthy":"unhealthy","unhealthy_prob":"unhealthy","prob_unhealthy":"unhealthy",
    "rubbish":"rubbish","rubbish_prob":"rubbish","prob_rubbish":"rubbish",
    "bothcells":"bothcells","bothcells_prob":"bothcells",
}
SEED=42
torch.manual_seed(SEED); np.random.seed(SEED)

def norm(n):
    n=n.strip(); return n[:-4] if n.lower().endswith(".png") else n

def file_map(data_dir):
    d=Path(data_dir)
    return {
        "YMG":{"test":d/"wrapup_UT/wrapup_UT/test_phase_prob.csv","eval":d/"wrapup_UT/wrapup_UT/final_eval_phase_revised_prob.csv"},
        "JNG":{"test":d/"jianght_challenge_materials/Evaluation-set.csv","eval":d/"jianght_challenge_materials/Test-set.csv"},
        "NGU":{"test":d/"NGU/predictions_isbi2025-ps3c-test-dataset.csv","eval":d/"NGU/predictions_isbi2025-ps3c-eval-dataset.csv"},
        "CHA":{"test":d/"ChatoLina_challenge_materials/isbi2025-ps3c-test-dataset pro Ens.csv","eval":d/"ChatoLina_challenge_materials/isbi2025-ps3c-eval-dataset pro Ens.csv"},
        "GUP":{"test":d/"Chinmay materials/ISBI PS3C IIT KGP/Test_Set_ProbabilityScores.csv","eval":d/"Chinmay materials/ISBI PS3C IIT KGP/Eval_Set_ProbabilityScores.csv"},
        "DPZ":{"test":d/"Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_test.csv","eval":d/"Re_ [PS3C - Joint paper] Materials - DI PIAZZA Theo/probabilities_eval.csv"},
        "WAN":{"test":d/"WAN/validation_predictions2.csv","eval":d/"WAN/val_predictions_resnet50.csv"},
    }

def load_labels(labels_dir,split):
    path=Path(labels_dir)/f"isbi2025-ps3c-{split}-dataset-annotated.csv"; gt={}
    with open(path,newline="",encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab=row["label"].strip().lower()
            if lab in CANON_IDX: gt[norm(row["image_name"])]=CANON_IDX[lab]
    return gt

def load_team(path,team):
    with open(path,newline="",encoding="utf-8") as f:
        rd=csv.reader(f); header=[h.strip().lower() for h in next(rd)]
        rows=[r for r in rd if r and r[0].strip()]
    d={}
    if team=="DPZ":
        ri,hi,ui=header.index("rubbish"),header.index("healthy"),header.index("unhealthy")
        for row in rows:
            nm=norm(row[0]); rub,hea,unh=float(row[ri]),float(row[hi]),float(row[ui])
            v=np.zeros(3,dtype=np.float32); v[2 if rub>=DPZ_THR else (0 if hea>=unh else 1)]=1.0; d[nm]=v
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

def load_all(data_dir,labels_dir,split):
    gt=load_labels(labels_dir,split); fm=file_map(data_dir)
    pt={t:load_team(fm[t][split],t) for t in TEAMS}
    common=set(gt)
    for t in TEAMS: common&=set(pt[t])
    names=sorted(common)
    probs=np.stack([np.array([pt[t][n] for n in names],dtype=np.float32) for t in TEAMS],axis=1)
    labels=np.array([gt[n] for n in names],dtype=np.int64)
    return probs,labels

def rank_avg_f1(probs,labels):
    N,T,C=probs.shape; ranks=np.zeros_like(probs)
    for t in range(T):
        for c in range(C):
            order=np.argsort(probs[:,t,c]); ranks[order,t,c]=np.arange(N)
    return float(f1_score(labels,ranks.mean(1).argmax(1),average="weighted"))


class WeightGeneratorCluster(nn.Module):
    """
    Harangi's cluster-based Weight Generator.

    Feature Encoder  → compact latent embedding from 21-dim prob vector
    Cluster Prototypes → K learnable centres in latent space
    Per-Cluster Weights → each cluster trusts different teams
    Aggregator → nonlinear fusion with residual connection
    """
    def __init__(self, n_teams=7, n_classes=3, n_clusters=8, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.n_clusters=n_clusters
        in_dim=n_teams*n_classes

        self.encoder=nn.Sequential(
            nn.Linear(in_dim,hidden_dim*2), nn.BatchNorm1d(hidden_dim*2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*2,hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
        )
        self.prototypes   = nn.Parameter(torch.randn(n_clusters,hidden_dim)/(hidden_dim**0.5))
        self.cluster_w    = nn.Parameter(torch.zeros(n_clusters,n_teams))
        self.aggregator   = nn.Sequential(
            nn.Linear(n_classes,n_classes*4), nn.GELU(), nn.Linear(n_classes*4,n_classes))

    def forward(self, team_probs):
        B,T,C=team_probs.shape
        h = self.encoder(team_probs.reshape(B,T*C))
        sim     = F.normalize(h,dim=-1) @ F.normalize(self.prototypes,dim=-1).T
        routing = F.softmax(sim*5.0,dim=-1)
        team_w  = routing @ F.softmax(self.cluster_w,dim=-1)
        fused   = (team_w.unsqueeze(-1)*team_probs).sum(1)
        logits  = fused + self.aggregator(fused)*0.1
        return logits, routing, team_w


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data-dir",   required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--out-dir",    default=".")
    parser.add_argument("--n-clusters", type=int,   default=8)
    parser.add_argument("--hidden-dim", type=int,   default=64)
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch-size", type=int,   default=256)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--patience",   type=int,   default=15)
    args=parser.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Clusters: {args.n_clusters}")

    print("\nLoading data...")
    probs_tr,labels_tr=load_all(args.data_dir,args.labels_dir,"test")
    probs_ev,labels_ev=load_all(args.data_dir,args.labels_dir,"eval")

    # Baselines
    ra_ev=rank_avg_f1(probs_ev,labels_ev)
    sa_ev=float(f1_score(labels_ev,probs_ev.mean(1).argmax(1),average="weighted"))
    print(f"Rank Averaging (hold-out): {ra_ev:.4f}")
    print(f"Simple Average (hold-out): {sa_ev:.4f}")

    # Train/val split on test set
    idx=np.arange(len(labels_tr))
    ix_tr,ix_val=train_test_split(idx,test_size=0.2,random_state=SEED,stratify=labels_tr)

    def tt(a, long=False):
        dt=torch.long if long else torch.float32
        return torch.from_numpy(a).to(dt).to(device)

    X_tr=tt(probs_tr[ix_tr]); y_tr=tt(labels_tr[ix_tr],True)
    X_val=tt(probs_tr[ix_val]); y_val=tt(labels_tr[ix_val],True)
    X_ev=tt(probs_ev)

    print(f"Train: {len(y_tr)}  Val: {len(y_val)}  Eval: {len(labels_ev)}")

    model=WeightGeneratorCluster(n_clusters=args.n_clusters,hidden_dim=args.hidden_dim).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    crit=nn.CrossEntropyLoss()

    best_val,best_state,no_imp=-1.0,None,0
    print(f"\nTraining ({args.epochs} epochs, patience={args.patience})...")

    for ep in range(1,args.epochs+1):
        model.train()
        perm=torch.randperm(len(y_tr)); X_tr,y_tr=X_tr[perm],y_tr[perm]
        tot_loss,nb=0.0,0
        for s in range(0,len(y_tr),args.batch_size):
            xb,yb=X_tr[s:s+args.batch_size],y_tr[s:s+args.batch_size]
            logits,routing,_=model(xb)
            loss=crit(logits,yb)
            ent=-(routing*(routing+1e-8).log()).sum(-1).mean()
            loss=loss+0.01*ent
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot_loss+=loss.item(); nb+=1
        sch.step()

        model.eval()
        with torch.no_grad():
            val_logits,_,_=model(X_val)
        val_preds=val_logits.argmax(1).cpu().numpy()
        val_f1=float(f1_score(y_val.cpu().numpy(),val_preds,average="weighted"))

        if val_f1>best_val:
            best_val=val_f1; best_state={k:v.clone() for k,v in model.state_dict().items()}
            no_imp=0; marker=" ← best"
        else:
            no_imp+=1; marker=f" (no improve {no_imp}/{args.patience})"

        if ep%10==0 or ep<=5:
            print(f"  Epoch {ep:3d}/{args.epochs}  loss={tot_loss/nb:.4f}  val_wF1={val_f1:.4f}{marker}")

        if no_imp>=args.patience:
            print(f"\n  Early stop at epoch {ep}"); break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        ev_logits,routing_ev,team_w_ev=model(X_ev)
    preds_ev=ev_logits.argmax(1).cpu().numpy()
    final_f1=float(f1_score(labels_ev,preds_ev,average="weighted"))

    print("\n"+"="*60)
    print("RESULTS — hold-out eval set")
    print("="*60)
    arrow="▲" if final_f1>ra_ev else "▼"
    print(f"  Rank Averaging baseline:    {ra_ev:.4f}")
    print(f"  WeightGenerator (cluster):  {final_f1:.4f}  {arrow} {final_f1-ra_ev:+.4f} vs rank avg")
    print(f"  Best val wF1 during train:  {best_val:.4f}")
    print()
    print("Per-class report:")
    print(classification_report(labels_ev,preds_ev,
          target_names=["healthy","unhealthy","rubbish"],digits=4))

    # Cluster analysis
    routing_np=routing_ev.cpu().numpy(); team_w_np=team_w_ev.cpu().numpy()
    assignments=routing_np.argmax(1)
    cw=F.softmax(model.cluster_w.detach(),dim=-1).cpu().numpy()
    print("Cluster team weights & dominant class:")
    print(f"  {'Cluster':<9}", end="")
    for t in TEAMS: print(f" {t:>5}", end="")
    print(f"  {'Size':>6}  Dom.Class")
    for k in range(args.n_clusters):
        mask=assignments==k; sz=mask.sum()
        dc=["healthy","unhealthy","rubbish"][int(np.bincount(labels_ev[mask]).argmax())] if sz>0 else "—"
        ws=" ".join(f"{w:.3f}" for w in cw[k])
        print(f"  {k:<9} {ws}  {sz:>6}  {dc}")

    print("\nMean team weight (all eval):")
    for t,w in sorted(zip(TEAMS,team_w_np.mean(0)),key=lambda x:-x[1]):
        print(f"  {t}: {w:.4f}  {'█'*int(w*40)}")

    # Save
    torch.save(model.state_dict(), out/"weight_generator_cluster.pt")
    np.save(out/"cluster_routing_eval.npy", routing_np)
    summary={"rank_averaging":ra_ev,"weight_generator_cluster":final_f1,
             "delta_vs_rank_avg":final_f1-ra_ev,"n_clusters":args.n_clusters}
    (out/"weight_generator_results.json").write_text(json.dumps(summary,indent=2))
    print(f"\nSaved model → {out}/weight_generator_cluster.pt")
    print(f"Saved results → {out}/weight_generator_results.json")
    print("="*60)

if __name__=="__main__":
    main()
