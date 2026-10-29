"""Probe: Dawid-Skene (two-coin) label-free reliability for evidence families."""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sklearn.metrics import roc_auc_score
from src.evidence.ground_truth import load_benchmark
from src.evidence import profilers as P
from src.evidence.fusion import load_llm_evidence, weight, masses

def dawid_skene(V, iters=100):
    q = V.mean(1)
    for _ in range(iters):
        pi = np.clip(q.mean(), 1e-4, 1-1e-4)
        sens = np.clip((q[:,None]*V).sum(0)/q.sum(), 1e-3, 1-1e-3)
        spec = np.clip(((1-q)[:,None]*(1-V)).sum(0)/(1-q).sum(), 1e-3, 1-1e-3)
        lo = np.log(pi/(1-pi)) + (V*np.log(sens/(1-spec)) + (1-V)*np.log((1-sens)/spec)).sum(1)
        q = 1/(1+np.exp(-lo))
    return q, pi, sens, spec

def prf(pred, gt):
    tp=(pred&gt).sum(); fp=(pred&~gt).sum(); fn=(~pred&gt).sum()
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return round(p,3), round(r,3), round(2*p*r/(p+r) if p+r else 0,3)

RUNS = Path(__file__).resolve().parents[2]/"experiments"/"runs"
for ds in ["hospital","flights","beers","rayyan"]:
    b=load_benchmark(ds); t=b.splits["test"]; n,m=b.errors.shape; gt=b.errors[t]; yrow=gt.any(1)
    ev=P.run_profilers(b.dirty)
    llm=load_llm_evidence(RUNS/f"auto_{ds}_test_full"/"per_sample.jsonl", n, b.columns)
    fam = {"syntax": np.max([ev[k] for k in ev if P.FAMILY[k]=="syntax"],0),
           "value": np.max([ev[k] for k in ev if P.FAMILY[k]=="value"],0)}
    llm_cell = 1-np.exp(-np.maximum(llm.cell_pos["auditor"], llm.cell_pos["expert"]))
    # cell-level EM over test cells with 3 families
    Vc = np.stack([fam["syntax"][t].ravel()>=0.5, fam["value"][t].ravel()>=0.5, llm_cell[t].ravel()>0], 1).astype(float)
    q, pi, sens, spec = dawid_skene(Vc)
    ppv = sens*pi/(sens*pi+(1-spec)*(1-pi))
    alpha = np.clip((ppv-pi)/(1-pi),0,1)
    # discounted fusion (family rule) with EM reliabilities
    wpos = alpha[0]*0 + weight(alpha[0]*fam["syntax"][t]) + weight(alpha[1]*fam["value"][t]) + weight(alpha[2]*llm_cell[t])
    wneg = np.repeat(np.max([llm.row_neg[a][t] for a in llm.row_neg],0)[:,None], m, 1)
    cm = masses(wpos, wneg)
    print(f"== {ds}: pi={pi:.3f} sens={np.round(sens,2)} spec={np.round(spec,3)} ppv={np.round(ppv,2)} alpha={np.round(alpha,2)}")
    print("   EM posterior>=.5 cell P/R/F1", prf(q.reshape(gt.shape)>=0.5, gt), "| discounted fusion m_err>=.5", prf(cm.m_err>=0.5, gt))
    # row-level EM with 4 annotators: syntax, value (row max), auditor, expert verdicts
    Vr = np.stack([fam["syntax"][t].max(1)>=0.5, fam["value"][t].max(1)>=0.5, llm.row_pos["auditor"][t]>0, llm.row_pos["expert"][t]>0],1).astype(float)
    qr, pir, sr, spr = dawid_skene(Vr)
    ppvr = sr*pir/(sr*pir+(1-spr)*(1-pir)); npvr = spr*(1-pir)/(spr*(1-pir)+(1-sr)*pir)
    ar = np.clip((ppvr-pir)/(1-pir),0,1); an = np.clip((npvr-(1-pir))/pir,0,1)
    wp = weight(ar[0]*fam["syntax"][t].max(1)) + weight(ar[1]*fam["value"][t].max(1)) + np.maximum(ar[2]*llm.row_pos["auditor"][t], ar[3]*llm.row_pos["expert"][t])
    wn = np.maximum(an[2]*llm.row_neg["auditor"][t], an[3]*llm.row_neg["expert"][t])
    rm = masses(wp, wn)
    if 0<yrow.mean()<1:
        print(f"   row: pi={pir:.2f} sens={np.round(sr,2)} spec={np.round(spr,2)} a+={np.round(ar,2)} a-={np.round(an,2)}  AUROC EMpost={roc_auc_score(yrow,qr):.3f} discounted={roc_auc_score(yrow,rm.score):.3f}")
