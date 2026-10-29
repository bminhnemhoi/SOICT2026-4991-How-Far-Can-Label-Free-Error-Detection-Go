"""Additional statistics requested in review round 2 (all from cached outputs)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "rivf"))
from run_experiments import raha_cells  # noqa: E402
from src.evidence import profilers as P  # noqa: E402
from src.evidence.fusion import apply_budget, budget_threshold, fuse, load_llm_evidence  # noqa: E402
from src.evidence.ground_truth import load_benchmark  # noqa: E402

RUNS = ROOT / "experiments" / "runs"
rng = np.random.default_rng(7)
B = 2000
out: dict = {}
for ds in ("hospital", "flights", "beers", "rayyan"):
    b = load_benchmark(ds)
    n, m = b.errors.shape
    t = b.splits["test"]
    gt = b.errors[t]
    y = gt.any(1)
    ev = P.run_profilers(b.dirty)
    llm = load_llm_evidence(RUNS / f"auto_{ds}_test_full" / "per_sample.jsonl", n, b.columns)
    prof = fuse(ev, P.FAMILY, None, rule="family", use_llm=False)
    full = fuse(ev, P.FAMILY, llm, rule="family", use_llm=True)
    ours = prof["cell"].m_err[t] >= 0.5
    rm = [M[t] for M in raha_cells(ds, (n, m), b.raw_col_index)]
    R: dict = {}

    # paired cell-level F1 difference vs RAHA (mean over seeds), row bootstrap
    def f1(pred, g):
        tp = (pred & g).sum(); fp = (pred & ~g).sum(); fn = (~pred & g).sum()
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    obs = f1(ours, gt) - np.mean([f1(M, gt) for M in rm])
    diffs = []
    for _ in range(B):
        idx = rng.integers(0, len(t), len(t))
        diffs.append(f1(ours[idx], gt[idx]) - np.mean([f1(M[idx], gt[idx]) for M in rm]))
    R["cell_dF1_vs_raha"] = {"diff": float(obs), "CI": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]}

    if 0 < y.mean() < 1:
        s = full["row"].score[t]
        # paired AUROC difference vs each RAHA seed
        R["row_auroc_vs_raha_seeds"] = []
        for M in rm:
            rs = M.mean(1)
            d = []
            for _ in range(B):
                idx = rng.integers(0, len(t), len(t))
                if 0 < y[idx].mean() < 1:
                    d.append(roc_auc_score(y[idx], s[idx]) - roc_auc_score(y[idx], rs[idx]))
            R["row_auroc_vs_raha_seeds"].append({"raha": float(roc_auc_score(y, rs)),
                                                 "diff": float(roc_auc_score(y, s) - roc_auc_score(y, rs)),
                                                 "CI": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]})
        # exact top-10% conflict rows
        K = full["row"].K[t]
        order = np.lexsort((rng.random(len(K)), -K))
        top = np.zeros(len(K), dtype=bool); top[order[: int(round(0.1 * len(K)))]] = True
        R["conflict_top10"] = {"err_rate_top": float(y[top].mean()), "err_rate_rest": float(y[~top].mean())}
        # distinct values of RAHA row score
        R["raha_row_score_distinct"] = [int(len(np.unique(M.mean(1)))) for M in rm]

    # budget: probability that the realised load exceeds beta + 0.05
    s_all = full["row"].score[t]
    over = {}
    for beta in (0.1, 0.2, 0.3, 0.4, 0.5):
        cnt = 0
        for _ in range(500):
            perm = rng.permutation(len(t)); h = len(t) // 2
            thr = budget_threshold(s_all[perm[:h]], beta, rng)
            cnt += apply_budget(s_all[perm[h:]], thr, rng).mean() > beta + 0.05
        over[f"{beta:.1f}"] = cnt / 500
    R["P_load_exceeds_beta_plus_0.05"] = over
    out[ds] = R
    print(ds, json.dumps(R), flush=True)

(ROOT / "experiments" / "rivf2026" / "extra_stats.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
