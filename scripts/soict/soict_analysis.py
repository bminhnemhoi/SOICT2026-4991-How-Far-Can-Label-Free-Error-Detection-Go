"""Analyses added for the SOICT 2026 camera-ready (reviewer requests).

  R1a  judge-related choices re-made on the dev rows (gpt-5-nano dev outputs)
  R1b  value of the evidential fusion: DS row score vs. label-free aggregators
       of the same inputs, plus a supervised stacking reference (uses labels)
  R1c  robustness across tables: a 20-tuple routing check between the
       label-free profilers and RAHA-20, on all six tables
  R2   judges from another model family (Claude Haiku 5.5) and a larger model
       of the same family (gpt-5-mini)

Writes experiments/soict2026/soict_results.json.  No LLM calls.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "rivf"))
from run_experiments import raha_cells  # noqa: E402
from src.evidence import profilers as P  # noqa: E402
from src.evidence.fusion import excess_strength, fuse, load_llm_evidence, weight  # noqa: E402
from src.evidence.ground_truth import load_benchmark  # noqa: E402

RUNS = ROOT / "experiments" / "runs"
OUT = ROOT / "experiments" / "soict2026"
RAW = ROOT / "data" / "raw" / "raha" / "datasets"
B = 1000
rng = np.random.default_rng(2026)

LLMS = {  # name -> run directory pattern for the test rows
    "gpt-5-nano": "auto_{ds}_test_full",
    "claude-haiku-5-5": "soict_claudehaiku55_{ds}_test",
    "gpt-5-mini": "soict_gpt5mini_{ds}_test",
}
EXTRA_LLMS = {  # added later, analysed with a separate random stream
    "gemini-3.8-flash": "soict_aggemini38flashlow_{ds}_test",
    "qwen3.8-27b": "soict_LamngQwenQwen3827BFP8_{ds}_test",
}
STRENGTHS = {
    "excess max(0,2p-1)": excess_strength,
    "raw p": lambda p: p,
    "verdict only (0.5)": lambda p: 0.5,
}


def prf(pred, gt):
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0}


def auc(y, s):
    return float(roc_auc_score(y, s))


def auc_ci(y, s):
    vals = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].mean() < 1:
            vals.append(roc_auc_score(y[i], s[i]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def paired(y, s1, s2):
    d = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].mean() < 1:
            d.append(roc_auc_score(y[i], s1[i]) - roc_auc_score(y[i], s2[i]))
    return {"diff": auc(y, s1) - auc(y, s2), "CI": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def covered_rows(run: Path) -> np.ndarray:
    return np.array(sorted({json.loads(l)["row_id"] for l in run.open(encoding="utf-8")}), dtype=int)


def run_stats(run: Path) -> dict:
    tin = tout = 0
    for l in run.open(encoding="utf-8"):
        r = json.loads(l)
        tin += r.get("tokens_in", 0); tout += r.get("tokens_out", 0)
    return {"tokens_in": tin, "tokens_out": tout}


def row_scores(ev, llm):
    """Row-level inputs and the aggregators compared in R1b."""
    prof = fuse(ev, P.FAMILY, None, rule="family", use_llm=False)
    full = fuse(ev, P.FAMILY, llm, rule="family", use_llm=True)
    w_prof = -np.log(np.maximum(1 - prof["row"].m_err, 1e-12))  # W+ of the profilers (row)
    s_prof = 1 - np.exp(-w_prof)
    pa, pe = llm.p_err["auditor"], llm.p_err["expert"]
    cols = np.vstack([s_prof, pa, pe]).T
    return {
        "inputs": {"profilers": s_prof, "auditor p(err)": pa, "expert p(err)": pe},
        "feat": np.vstack([w_prof, llm.row_pos["auditor"], llm.row_neg["auditor"],
                           llm.row_pos["expert"], llm.row_neg["expert"]]).T,
        "DS fusion (ours)": full["row"].score,
        "mean": cols.mean(1),
        "max": cols.max(1),
        "noisy-OR": 1 - np.prod(1 - np.clip(cols, 0, 1), axis=1),
        "cols": cols,
        "judges only (DS)": fuse({k: np.zeros_like(v) for k, v in ev.items()}, P.FAMILY, llm,
                                 rule="family", use_llm=True)["row"].score,
        "profilers only": prof["row"].score,
    }


def stacking_cv(feat, y, seed=0):
    """Supervised reference: 5-fold cross-fitted logistic regression (uses labels)."""
    out = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(feat, y):
        out[te] = LogisticRegression(max_iter=1000).fit(feat[tr], y[tr]).predict_proba(feat[te])[:, 1]
    return out


def main() -> None:
    global rng
    OUT.mkdir(parents=True, exist_ok=True)
    res: dict = {"dev_choices": {}, "aggregators": {}, "models": {}, "routing": {}}
    for ds in ("hospital", "flights", "beers", "rayyan"):
        b = load_benchmark(ds)
        n, m = b.errors.shape
        ev = P.run_profilers(b.dirty)
        prof = fuse(ev, P.FAMILY, None, rule="family", use_llm=False)
        row_level = ds != "beers"  # every Beers row is erroneous

        # ---------- R1a: judge choices on dev rows ----------
        dev_run = RUNS / f"soict_gpt5nano_{ds}_dev" / "per_sample.jsonl"
        dv = b.splits["dev"]
        assert set(covered_rows(dev_run)) == set(dv.tolist()), f"{ds}: dev outputs incomplete"
        ydev, gdev = b.errors[dv].any(1), b.errors[dv]
        D: dict = {"n_dev": int(len(dv)), "row_prev": float(ydev.mean()), "strength_row_auroc": {}}
        for name, f in STRENGTHS.items():
            if not row_level:
                break
            llm = load_llm_evidence(dev_run, n, b.columns, strength=f)
            D["strength_row_auroc"][name] = auc(ydev, fuse(ev, P.FAMILY, llm, rule="family")["row"].score[dv])
            # the same alternative evaluated on the test rows, for transparency
            llm_t = load_llm_evidence(RUNS / f"auto_{ds}_test_full" / "per_sample.jsonl", n, b.columns, strength=f)
            tt = b.splits["test"]
            D.setdefault("strength_row_auroc_test", {})[name] = auc(
                b.errors[tt].any(1), fuse(ev, P.FAMILY, llm_t, rule="family")["row"].score[tt])
        llm = load_llm_evidence(dev_run, n, b.columns)
        full = fuse(ev, P.FAMILY, llm, rule="family", use_llm=True)
        D["cell_F1_profilers_only"] = prf(prof["cell"].m_err[dv] >= 0.5, gdev)["F1"]
        D["cell_F1_plus_judge_fields"] = prf(full["cell"].m_err[dv] >= 0.5, gdev)["F1"]
        D["cell_F1_judge_fields_only"] = prf(
            fuse({k: np.zeros_like(v) for k, v in ev.items()}, P.FAMILY, llm, rule="family")["cell"].m_err[dv] >= 0.5,
            gdev)["F1"]
        res["dev_choices"][ds] = D
        if not row_level:
            print(ds, "dev", json.dumps(D, default=float), flush=True)
            continue
        sc = row_scores(ev, llm)
        D["row_auroc_dev"] = {k: auc(ydev, sc[k][dv]) for k in ("DS fusion (ours)", "profilers only", "judges only (DS)")}
        D["row_auroc_dev_CI"] = {k: auc_ci(ydev, sc[k][dv]) for k in ("DS fusion (ours)",)}
        D["dev_gain_vs_profilers"] = paired(ydev, sc["DS fusion (ours)"][dv], sc["profilers only"][dv])
        res["dev_choices"][ds] = D
        print(ds, "dev", json.dumps({k: v for k, v in D.items()}, default=float)[:600], flush=True)

        # ---------- R2 + R1b: test rows, per LLM ----------
        res["models"][ds], res["aggregators"][ds] = {}, {}
        t_all = b.splits["test"]
        for name, pat in LLMS.items():
            run = RUNS / pat.format(ds=ds) / "per_sample.jsonl"
            if not run.exists():
                continue
            t = np.intersect1d(t_all, covered_rows(run))
            if len(t) < 100:
                continue
            y, g = b.errors[t].any(1), b.errors[t]
            llm = load_llm_evidence(run, n, b.columns)
            sc = row_scores(ev, llm)
            sc["rank average"] = np.full(n, np.nan)
            sc["rank average"][t] = np.mean([rankdata(c) for c in sc["cols"][t].T], axis=0)
            jf = fuse({k: np.zeros_like(v) for k, v in ev.items()}, P.FAMILY, llm, rule="family")["cell"].m_err[t] >= 0.5
            M = {"n_rows": int(len(t)), "complete": bool(len(t) == len(t_all)),
                 "judges_only": auc(y, sc["judges only (DS)"][t]),
                 "fused": auc(y, sc["DS fusion (ours)"][t]), "fused_CI": auc_ci(y, sc["DS fusion (ours)"][t]),
                 "profilers": auc(y, sc["profilers only"][t]),
                 "gain_vs_profilers": paired(y, sc["DS fusion (ours)"][t], sc["profilers only"][t]),
                 "gain_vs_judges": paired(y, sc["DS fusion (ours)"][t], sc["judges only (DS)"][t]),
                 "judge_field_cell_F1": prf(jf, g)["F1"],
                 "err_verdict_rate": float(np.mean([(llm.row_pos[a][t] > 0).mean() for a in llm.row_pos])),
                 **({} if name == "gpt-5-nano" else run_stats(run))}
            res["models"][ds][name] = M
            A = {k: auc(y, sc[k][t]) for k in ("DS fusion (ours)", "mean", "max", "noisy-OR", "rank average")}
            A["stacking (5-fold, labels)"] = float(np.mean([auc(y, stacking_cv(sc["feat"][t], y, s)) for s in range(5)]))
            best = max((k for k in ("mean", "max", "noisy-OR", "rank average")), key=lambda k: A[k])
            A["_best_simple"] = best
            A["_ds_vs_best_simple"] = paired(y, sc["DS fusion (ours)"][t], sc[best][t])
            res["aggregators"][ds][name] = A
            print(ds, name, json.dumps(M, default=float)[:300], "\n   agg", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in A.items() if not k.startswith("_ds")}, flush=True)

        # R1b: does the ignorance m(Omega) mark rows whose score is unreliable? (gpt-5-nano judges)
        llm = load_llm_evidence(RUNS / f"auto_{ds}_test_full" / "per_sample.jsonl", n, b.columns)
        row = fuse(ev, P.FAMILY, llm, rule="family", use_llm=True)["row"]
        y = b.errors[t_all].any(1)
        om, s = row.m_omega[t_all], row.score[t_all]
        hi = om >= np.quantile(om, 0.75)
        res.setdefault("ignorance", {})[ds] = {"q75": float(np.quantile(om, 0.75)),
                                               "auroc_top_quarter": auc(y[hi], s[hi]), "auroc_rest": auc(y[~hi], s[~hi])}
        print(ds, "ignorance", res["ignorance"][ds], flush=True)

        # three judges' families together (rows covered by all three)
        runs = [RUNS / p.format(ds=ds) / "per_sample.jsonl" for p in LLMS.values()]
        if all(r.exists() for r in runs):
            t = t_all
            for r in runs:
                t = np.intersect1d(t, covered_rows(r))
            if len(t) >= 100:
                y = b.errors[t].any(1)
                lls = [load_llm_evidence(r, n, b.columns) for r in runs]
                w_pos = -np.log(np.maximum(1 - prof["row"].m_err, 1e-12))
                w_neg = np.zeros(n)
                for L in lls:  # each model family is a distinct source: Dempster across, cautious within
                    w_pos = w_pos + np.maximum(L.row_pos["auditor"], L.row_pos["expert"])
                    w_neg = w_neg + np.maximum(L.row_neg["auditor"], L.row_neg["expert"])
                s3 = 1 / (1 + np.exp(-(w_pos - w_neg)))
                s1 = row_scores(ev, lls[0])["DS fusion (ours)"]
                res["models"][ds]["all judges"] = {"n_rows": int(len(t)), "fused": auc(y, s3[t]),
                                                      "vs_gpt5nano": paired(y, s3[t], s1[t])}

    # ---------- R1c: 20-tuple routing between profilers and RAHA-20 ----------
    held = json.loads((ROOT / "experiments" / "rivf2026" / "heldout" / "heldout.json").read_text())
    tables = {}
    for ds in ("hospital", "flights", "beers", "rayyan"):
        b = load_benchmark(ds)
        n, m = b.errors.shape
        ev = P.run_profilers(b.dirty)
        pred = fuse(ev, P.FAMILY, None, rule="family", use_llm=False)["cell"].m_err >= 0.5
        raha = float(np.mean([prf(M, b.errors)["F1"] for M in raha_cells(ds, (n, m), b.raw_col_index)]))
        tables[ds] = (pred, b.errors, raha)
    for ds, spec in {"movies_1": ({"id"}, None), "tax": (set(), 20000)}.items():
        dirty = pd.read_csv(RAW / ds / "dirty.csv", dtype=str, keep_default_na=False, nrows=spec[1])
        clean = pd.read_csv(RAW / ds / "clean.csv", dtype=str, keep_default_na=False, nrows=spec[1])
        keep = [i for i, c in enumerate(dirty.columns) if c.lower() not in spec[0]]
        Dd = dirty.iloc[:, keep].reset_index(drop=True)
        gt = Dd.to_numpy() != clean.iloc[:, keep].reset_index(drop=True).to_numpy()
        pred = fuse(P.run_profilers(Dd), P.FAMILY, None, rule="family", use_llm=False)["cell"].m_err >= 0.5
        tables[ds] = (pred, gt, held[ds]["RAHA-20"]["F1_mean"])
    for ds, (pred, gt, raha) in tables.items():
        f_prof = prf(pred, gt)["F1"]
        picks = []
        for _ in range(1000):
            idx = rng.choice(len(gt), 20, replace=False)
            pp, gg = pred[idx], gt[idx]
            est = prf(pp, gg)["F1"] if (pp.any() or gg.any()) else 1.0  # nothing to see: keep profilers
            picks.append(est >= 0.5)
        picks = np.array(picks)
        res["routing"][ds] = {"F1_profilers": f_prof, "F1_raha20": raha,
                              "better": "profilers" if f_prof >= raha else "RAHA-20",
                              "P_keep_profilers": float(picks.mean()),
                              "expected_F1_routed": float(picks.mean() * f_prof + (1 - picks.mean()) * raha),
                              "P_correct_choice": float(picks.mean() if f_prof >= raha else 1 - picks.mean())}
        print("routing", ds, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["routing"][ds].items()}, flush=True)

    # ---------- R2 (extension): two more judge families and run-to-run variation ----------
    # A separate random stream keeps every number above unchanged.
    rng = np.random.default_rng(4991)
    res["rerun"] = {}
    for ds in ("hospital", "flights", "rayyan"):
        b = load_benchmark(ds)
        n, m = b.errors.shape
        ev = P.run_profilers(b.dirty)
        t_all = b.splits["test"]
        for name, pat in EXTRA_LLMS.items():
            run = RUNS / pat.format(ds=ds) / "per_sample.jsonl"
            if not run.exists():
                continue
            t = np.intersect1d(t_all, covered_rows(run))
            if len(t) < 100:
                continue
            y, g = b.errors[t].any(1), b.errors[t]
            llm = load_llm_evidence(run, n, b.columns)
            sc = row_scores(ev, llm)
            jf = fuse({k: np.zeros_like(v) for k, v in ev.items()}, P.FAMILY, llm, rule="family")["cell"].m_err[t] >= 0.5
            res["models"][ds][name] = {
                "n_rows": int(len(t)), "complete": bool(len(t) == len(t_all)),
                "judges_only": auc(y, sc["judges only (DS)"][t]),
                "fused": auc(y, sc["DS fusion (ours)"][t]), "fused_CI": auc_ci(y, sc["DS fusion (ours)"][t]),
                "profilers": auc(y, sc["profilers only"][t]),
                "gain_vs_profilers": paired(y, sc["DS fusion (ours)"][t], sc["profilers only"][t]),
                "gain_vs_judges": paired(y, sc["DS fusion (ours)"][t], sc["judges only (DS)"][t]),
                "judge_field_cell_F1": prf(jf, g)["F1"],
                "err_verdict_rate": float(np.mean([(llm.row_pos[a][t] > 0).mean() for a in llm.row_pos])),
                **run_stats(run)}
            print(ds, name, {k: v for k, v in res["models"][ds][name].items() if k in ("n_rows", "judges_only", "fused")}, flush=True)
        # Claude Haiku 5.5 judges queried a second time (new request seed, no cache)
        r1 = RUNS / f"soict_claudehaiku55_{ds}_test" / "per_sample.jsonl"
        r2 = RUNS / f"soict_claudehaiku55_{ds}_test_s43" / "per_sample.jsonl"
        if r2.exists():
            t = np.intersect1d(t_all, covered_rows(r2))
            if len(t) >= 100:
                y = b.errors[t].any(1)
                L1, L2 = load_llm_evidence(r1, n, b.columns), load_llm_evidence(r2, n, b.columns)
                s1, s2 = row_scores(ev, L1), row_scores(ev, L2)
                agree = float(np.mean([((L1.row_pos[a][t] > 0) == (L2.row_pos[a][t] > 0)).mean() for a in L1.row_pos]))
                res["rerun"][ds] = {"n_rows": int(len(t)), "verdict_agreement": agree,
                                    "judges_only": [auc(y, s1["judges only (DS)"][t]), auc(y, s2["judges only (DS)"][t])],
                                    "fused": [auc(y, s1["DS fusion (ours)"][t]), auc(y, s2["DS fusion (ours)"][t])],
                                    "fused_diff": paired(y, s2["DS fusion (ours)"][t], s1["DS fusion (ours)"][t])}
                print(ds, "rerun", res["rerun"][ds], flush=True)

    (OUT / "soict_results.json").write_text(json.dumps(res, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
