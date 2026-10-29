"""All RIVF 2026 experiments from one script (no new LLM calls).

Outputs experiments/rivf2026/results.json and prints the paper tables.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.evidence import profilers as P  # noqa: E402
from src.evidence.fusion import (  # noqa: E402
    apply_budget, budget_threshold, fuse, load_llm_evidence, weight,
)
from src.evidence.ground_truth import load_benchmark  # noqa: E402

RUNS = ROOT / "experiments" / "runs"
OUT = ROOT / "experiments" / "rivf2026"
DATASETS = ["hospital", "flights", "beers", "rayyan"]
B_BOOT = 1000
RNG = np.random.default_rng(2026)


def prf(pred: np.ndarray, gt: np.ndarray) -> dict:
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"P": p, "R": r, "F1": f, "tp": tp, "fp": fp, "fn": fn}


def boot_f1(pred: np.ndarray, gt: np.ndarray, B: int = B_BOOT) -> tuple[float, float]:
    """Row-bootstrap 95% CI of cell-level F1 (pred, gt are n x m)."""
    n = len(pred)
    tp = (pred & gt).sum(1); fp = (pred & ~gt).sum(1); fn = (~pred & gt).sum(1)
    vals = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        T, F, N = tp[idx].sum(), fp[idx].sum(), fn[idx].sum()
        vals.append(2 * T / (2 * T + F + N) if T else 0.0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def auc_ci(y: np.ndarray, s: np.ndarray, B: int = B_BOOT) -> dict:
    out = {"AUROC": float(roc_auc_score(y, s)), "AUPRC": float(average_precision_score(y, s))}
    n = len(y); v = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        if 0 < y[idx].mean() < 1:
            v.append(roc_auc_score(y[idx], s[idx]))
    out["AUROC_CI"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return out


def paired_auc_diff(y: np.ndarray, s1: np.ndarray, s2: np.ndarray, B: int = B_BOOT) -> dict:
    n = len(y); d = []
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        if 0 < y[idx].mean() < 1:
            d.append(roc_auc_score(y[idx], s1[idx]) - roc_auc_score(y[idx], s2[idx]))
    return {"diff": float(roc_auc_score(y, s1) - roc_auc_score(y, s2)),
            "CI": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_le0": float(np.mean(np.array(d) <= 0))}


def iforest_scores(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Label-free Isolation Forest on frequency-encoded + shape-encoded columns."""
    feats = []
    for c in df.columns:
        feats.append(df[c].map(df[c].value_counts(normalize=True)).to_numpy())
        sh = df[c].map(P.shape)
        feats.append(sh.map(sh.value_counts(normalize=True)).to_numpy())
        feats.append(df[c].str.len().to_numpy() / max(1, df[c].str.len().max()))
    X = np.column_stack(feats)
    clf = IsolationForest(n_estimators=300, random_state=seed).fit(X)
    return -clf.score_samples(X)


def load_chat_scores(ds: str, n: int) -> np.ndarray:
    s = np.full(n, np.nan)
    f = RUNS / f"anollm_{ds}_full" / "per_sample.jsonl"
    if f.exists():
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            s[int(r["row_id"])] = float(r["score"])
    return s


def load_v11_auto(ds: str, n: int) -> np.ndarray:
    s = np.full(n, np.nan)
    for line in open(RUNS / f"auto_{ds}_test_full" / "per_sample.jsonl", encoding="utf-8"):
        r = json.loads(line)
        s[int(r["row_id"])] = float(r["ds"]["bel_err"])
    return s


def llm_cost(ds: str, n_rows: int) -> dict:
    s = json.load(open(RUNS / f"auto_{ds}_test_full" / "summary.json"))
    c = s["cost"]
    per_call = c["usd"] / max(1, c["n_calls"])
    return {"logged_usd": c["usd"], "n_calls_billed": c["n_calls"], "cache_hits": c["n_cache_hits"],
            "est_usd_full": per_call * 2 * n_rows,
            "tokens_in_per_call": c["tokens_in"] / max(1, c["n_calls"])}


def raha_cells(ds: str, shape: tuple[int, int], raw_col_index: list[int]) -> list[np.ndarray]:
    """RAHA indexes columns of dirty.csv; flags on identifier columns (which
    contain no errors) are dropped, which can only help RAHA."""
    to_data = {raw: j for j, raw in enumerate(raw_col_index)}
    mats = []
    for f in sorted((OUT / "raha").glob(f"{ds}_seed*.json")):
        M = np.zeros(shape, dtype=bool)
        for i, j in json.load(open(f))["cells"]:
            if j in to_data:
                M[i, to_data[j]] = True
        mats.append(M)
    return mats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results: dict = {}
    for ds in DATASETS:
        t0 = time.time()
        b = load_benchmark(ds)
        n, m = b.errors.shape
        test = b.splits["test"]
        gt = b.errors[test]
        yrow = gt.any(1)
        ev = P.run_profilers(b.dirty)
        fds = P.mine_fds(b.dirty)
        llm = load_llm_evidence(RUNS / f"auto_{ds}_test_full" / "per_sample.jsonl", n, b.columns)
        assert llm.covered[test].all(), f"{ds}: LLM outputs missing for some test rows"
        R: dict = {"n_test": int(len(test)), "m": m, "n_err_cells": int(gt.sum()),
                   "cell_err_rate": float(gt.mean()), "row_prev": float(yrow.mean()),
                   "n_fds": len(fds), "llm_cost": llm_cost(ds, len(test))}

        # ---------------- cell level ----------------
        cell: dict = {}
        for k, E in ev.items():
            cell[f"P:{k}"] = prf(E[test] >= 0.5, gt)
        zero = {k: np.zeros_like(v) for k, v in ev.items()}
        variants = {
            "profilers(family)": fuse(ev, P.FAMILY, llm, rule="family", use_llm=False),
            "llm-only": fuse(zero, P.FAMILY, llm, rule="family", use_llm=True),
            "full(family)": fuse(ev, P.FAMILY, llm, rule="family", use_llm=True),
            "full(dempster)": fuse(ev, P.FAMILY, llm, rule="dempster", use_llm=True),
            "full(cautious)": fuse(ev, P.FAMILY, llm, rule="cautious", use_llm=True),
        }
        for name, fz in variants.items():
            pred = fz["cell"].m_err[test] >= 0.5
            cell[name] = prf(pred, gt)
            if name in ("full(family)", "profilers(family)", "llm-only"):
                cell[name]["F1_CI"] = boot_f1(pred, gt)
        # ablations of the cell detector (profilers, no LLM)
        for fam in ("syntax", "value"):
            ev_minus = {k: (np.zeros_like(v) if P.FAMILY[k] == fam else v) for k, v in ev.items()}
            fz = fuse(ev_minus, P.FAMILY, None, rule="family", use_llm=False)
            cell[f"prof-minus-{fam}"] = prf(fz["cell"].m_err[test] >= 0.5, gt)
        for k in ev:
            ev_minus = {kk: (np.zeros_like(v) if kk == k else v) for kk, v in ev.items()}
            fz = fuse(ev_minus, P.FAMILY, None, rule="family", use_llm=False)
            cell[f"prof-minus-{k}"] = prf(fz["cell"].m_err[test] >= 0.5, gt)
        for rule in ("dempster", "cautious"):
            fz = fuse(ev, P.FAMILY, None, rule=rule, use_llm=False)
            cell[f"prof({rule})"] = prf(fz["cell"].m_err[test] >= 0.5, gt)
        rm = raha_cells(ds, (n, m), b.raw_col_index)
        if rm:
            f1s = [prf(M[test], gt) for M in rm]
            cell["RAHA-20"] = {kk: float(np.mean([x[kk] for x in f1s])) for kk in ("P", "R", "F1")}
            cell["RAHA-20"]["F1_sd"] = float(np.std([x["F1"] for x in f1s]))
            cell["RAHA-20"]["n_seeds"] = len(rm)
        R["cell"] = cell

        # cell decision from data evidence + LLM row verdicts as clean evidence (ablation)
        prof = variants["profilers(family)"]
        from src.evidence.fusion import masses as _masses
        w_prof = -np.log(np.maximum(1.0 - prof["cell"].m_err, 1e-6))
        row_neg = np.max([llm.row_neg[a] for a in llm.row_neg], axis=0)
        veto = _masses(w_prof, np.repeat(row_neg[:, None], m, axis=1))
        cell["profilers+LLM-clean-veto"] = prf(veto.m_err[test] >= 0.5, gt)

        # threshold sensitivity of the cell decision (main detector = profilers)
        full = variants["full(family)"]
        R["cell_threshold_sweep"] = {f"{t:.1f}": prf(prof["cell"].m_err[test] >= t, gt)["F1"]
                                     for t in (0.3, 0.4, 0.5, 0.6, 0.7)}

        # full-table evaluation of the label-free profilers (ZeroED/ForestED protocol)
        R["cell_full_table"] = prf(prof["cell"].m_err >= 0.5, b.errors)
        R["cell_dev"] = prf(prof["cell"].m_err[b.splits["dev"]] >= 0.5, b.errors[b.splits["dev"]])
        t1 = time.time(); P.run_profilers(b.dirty); R["profiler_seconds"] = time.time() - t1

        # Rayyan diagnostic: ground truth without the date column
        if ds == "rayyan":
            jd = b.columns.index("article_jcreated_at")
            keep = [j for j in range(m) if j != jd]
            gt_nd = gt[:, keep]
            R["rayyan_without_date"] = {
                "profilers": prf(prof["cell"].m_err[test][:, keep] >= 0.5, gt_nd),
                "llm-only": prf(variants["llm-only"]["cell"].m_err[test][:, keep] >= 0.5, gt_nd),
                "row_prev": float(gt_nd.any(1).mean()),
            }
            yr = gt_nd.any(1)
            R["rayyan_without_date"]["row_auroc"] = {
                "full(family)": float(roc_auc_score(yr, full["row"].score[test])),
                "profilers(family)": float(roc_auc_score(yr, prof["row"].score[test])),
                "llm-only": float(roc_auc_score(yr, variants["llm-only"]["row"].score[test])),
            }
            if rm:
                R["rayyan_without_date"]["RAHA-20"] = float(np.mean([prf(M[test][:, keep], gt_nd)["F1"] for M in rm]))

        # hyper-parameter sensitivity of the profilers (one parameter at a time)
        grid = {"PATTERN_RARE_SHARE": (0.02, 0.05, 0.10), "TYPO_MIN_RATIO": (80.0, 90.0, 95.0),
                "FD_MIN_CONF": (0.85, 0.90, 0.95), "FD_MIN_GROUP": (2, 3, 5)}
        sens = {}
        for par, vals in grid.items():
            old = getattr(P, par)
            for v in vals:
                setattr(P, par, v)
                ev_v = P.run_profilers(b.dirty)
                fz = fuse(ev_v, P.FAMILY, None, rule="family", use_llm=False)
                sens[f"{par}={v}"] = prf(fz["cell"].m_err[test] >= 0.5, gt)["F1"]
            setattr(P, par, old)
        R["sensitivity"] = sens

        # ---------------- row level ----------------
        if 0 < yrow.mean() < 1:
            scores = {
                "full(family)": full["row"].score[test],
                "full: bel(err)": full["row"].m_err[test],
                "profilers(family)": variants["profilers(family)"]["row"].score[test],
                "llm-only": variants["llm-only"]["row"].score[test],
                "v11 AUTO bel(err)": load_v11_auto(ds, n)[test],
                "chat LLM scorer": load_chat_scores(ds, n)[test],
                "IsolationForest": iforest_scores(b.dirty)[test],
            }
            if rm:
                scores["RAHA-20 (seed0)"] = rm[0][test].mean(1)
                R["raha_row_auroc_seeds"] = [float(roc_auc_score(yrow, M[test].mean(1))) for M in rm]
            row = {k: auc_ci(yrow, v) for k, v in scores.items() if not np.isnan(v).any()}
            best_base = max((k for k in row if k not in ("full(family)", "full: bel(err)")),
                            key=lambda k: row[k]["AUROC"])
            row["_paired_vs_best_baseline"] = {"baseline": best_base,
                                               **paired_auc_diff(yrow, scores["full(family)"], scores[best_base])}
            row["_paired_vs_llm_only"] = paired_auc_diff(yrow, scores["full(family)"], scores["llm-only"])
            row["_paired_vs_profilers"] = paired_auc_diff(yrow, scores["full(family)"], scores["profilers(family)"])
            row["_paired_vs_chat"] = paired_auc_diff(yrow, scores["full(family)"], scores["chat LLM scorer"])
            if rm:
                row["_paired_vs_raha"] = paired_auc_diff(yrow, scores["full(family)"], scores["RAHA-20 (seed0)"])
            R["row"] = row
            # review-budget curves: recall of erroneous rows when reviewing the top-beta fraction
            curves = {}
            betas = np.round(np.arange(0.05, 1.0001, 0.05), 2)
            for k, v in scores.items():
                if np.isnan(v).any() or k == "full: bel(err)":
                    continue
                rec = []
                for beta in betas:
                    kk = int(round(beta * len(v)))
                    reps = []
                    for _ in range(20):  # random tie-breaking
                        order = np.lexsort((RNG.random(len(v)), -v))
                        reps.append(yrow[order[:kk]].sum() / yrow.sum())
                    rec.append(float(np.mean(reps)))
                curves[k] = rec
            curves["oracle"] = [float(min(1.0, round(beta * len(yrow)) / yrow.sum())) for beta in betas]
            curves["random"] = [float(beta) for beta in betas]
            R["budget_curves"] = {"beta": betas.tolist(), **curves}

        # ---------------- conflict and review budget ----------------
        rowm = full["row"]
        Kt = rowm.K[test]
        hiK = Kt > np.quantile(Kt, 0.9) if Kt.max() > 0 else np.zeros_like(Kt, dtype=bool)
        R["conflict"] = {
            "K_max": float(Kt.max()), "K_mean": float(Kt.mean()),
            "frac_rows_K>0": float((Kt > 0).mean()),
            "err_rate_topdecile_K": float(yrow[hiK].mean()) if hiK.any() else None,
            "err_rate_rest": float(yrow[~hiK].mean()),
            "m_omega_mean": float(rowm.m_omega[test].mean()),
        }
        budget = {}
        s_all = rowm.score[test]
        for beta in (0.1, 0.2, 0.3, 0.4, 0.5):
            load, recall, rand_recall = [], [], []
            for _ in range(500):
                perm = RNG.permutation(len(test)); h = len(test) // 2
                cal, ev_idx = perm[:h], perm[h:]
                thr = budget_threshold(s_all[cal], beta, RNG)
                rev = apply_budget(s_all[ev_idx], thr, RNG)
                load.append(rev.mean())
                ye = yrow[ev_idx]
                recall.append(rev[ye].mean() if ye.any() else np.nan)
                rand_recall.append(rev.mean())
            budget[f"{beta:.1f}"] = {"load_mean": float(np.mean(load)), "load_sd": float(np.std(load)),
                                     "row_recall": float(np.nanmean(recall)),
                                     "bound": [beta - 1 / (len(test) // 2 + 1), beta]}
        R["budget"] = budget
        R["seconds"] = time.time() - t0
        results[ds] = R
        print(f"[{ds}] done in {R['seconds']:.0f}s", flush=True)

    (OUT / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # ---------------- print tables ----------------
    print("\nCELL-LEVEL F1 (test rows)")
    keys = ["P:pattern", "P:missing", "P:typo", "P:fd", "P:unit", "profilers(family)", "llm-only",
            "full(family)", "full(dempster)", "full(cautious)", "prof(dempster)", "prof(cautious)", "prof-minus-syntax", "prof-minus-value",
            "prof-minus-pattern", "prof-minus-missing", "prof-minus-typo", "prof-minus-fd", "prof-minus-unit",
            "profilers+LLM-clean-veto", "RAHA-20"]
    print(f"{'method':22s}" + "".join(f"{d:>22s}" for d in DATASETS))
    for k in keys:
        line = f"{k:22s}"
        for d in DATASETS:
            c = results[d]["cell"].get(k)
            line += f"{c['P']:.2f}/{c['R']:.2f}/{c['F1']:.3f}".rjust(22) if c else " " * 22
        print(line)
    print("\nROW-LEVEL AUROC [95% CI]")
    for d in DATASETS:
        if "row" not in results[d]:
            continue
        print(f"-- {d}")
        for k, v in results[d]["row"].items():
            if k.startswith("_"):
                print(f"   {k}: {v}")
            else:
                print(f"   {k:22s} AUROC={v['AUROC']:.3f} [{v['AUROC_CI'][0]:.3f},{v['AUROC_CI'][1]:.3f}] AUPRC={v['AUPRC']:.3f}")
    print("\nCONFLICT / BUDGET")
    for d in DATASETS:
        print(d, results[d]["conflict"])
        print("  ", {k: (round(v["load_mean"], 3), round(v["load_sd"], 3), round(v["row_recall"], 3)) for k, v in results[d]["budget"].items()})
        print("   cost", results[d]["llm_cost"], "thr-sweep", results[d]["cell_threshold_sweep"])
        print("   full-table", {k: round(v, 3) for k, v in results[d]["cell_full_table"].items() if k in "PRF1"},
              "dev", {k: round(v, 3) for k, v in results[d]["cell_dev"].items() if k in "PRF1"},
              "secs", round(results[d]["profiler_seconds"], 1))
        print("   sensitivity", {k: round(v, 3) for k, v in results[d]["sensitivity"].items()})
        if "RAHA-20" in results[d]["cell"]:
            print("   RAHA-20", results[d]["cell"]["RAHA-20"])
    print("rayyan without date:", json.dumps(results["rayyan"].get("rayyan_without_date"), indent=1))


if __name__ == "__main__":
    main()
