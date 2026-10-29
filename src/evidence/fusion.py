"""Evidential fusion on the frame {err, clean} in weight-of-evidence form.

A simple support function m({A}) = s, m(Omega) = 1 - s has weight of evidence
w = -ln(1 - s).  Dempster's rule adds weights of *distinct* sources; the
cautious rule (Denoeux, 2008) takes their maximum for *non-distinct* sources.
With total support W+ for err and W- for clean,

    S = 1 - exp(-W+),  T = 1 - exp(-W-),  K = S T,
    m(err)   = S (1 - T) / (1 - K),
    m(clean) = T (1 - S) / (1 - K),
    m(Omega) = (1 - S)(1 - T) / (1 - K),
    Pl(err) / (Pl(err) + Pl(clean)) = sigmoid(W+ - W-).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EPS = 1e-6
W_MAX = -np.log(EPS)


def weight(s: np.ndarray | float) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0 - EPS)
    return -np.log1p(-s)


@dataclass
class Masses:
    m_err: np.ndarray
    m_clean: np.ndarray
    m_omega: np.ndarray
    K: np.ndarray
    score: np.ndarray  # sigmoid(W+ - W-), the normalised plausibility of err


def masses(w_pos: np.ndarray, w_neg: np.ndarray) -> Masses:
    S = 1.0 - np.exp(-w_pos)
    T = 1.0 - np.exp(-w_neg)
    K = S * T
    d = np.maximum(1.0 - K, EPS)
    return Masses(
        m_err=S * (1.0 - T) / d,
        m_clean=T * (1.0 - S) / d,
        m_omega=(1.0 - S) * (1.0 - T) / d,
        K=K,
        score=1.0 / (1.0 + np.exp(-(w_pos - w_neg))),
    )


# --------------------------------------------------------------------------
# LLM evidence from cached per-row outputs
# --------------------------------------------------------------------------
@dataclass
class LLMEvidence:
    """Row- and cell-level weights from the Auditor and Expert agents.

    For agent a with verdict l in {err, clean} and probability p of that
    verdict, the strength is s = max(0, 2p - 1) (the excess over a coin flip).
    """

    row_pos: dict[str, np.ndarray]    # agent -> (n,) weight for err
    row_neg: dict[str, np.ndarray]    # agent -> (n,) weight for clean
    cell_pos: dict[str, np.ndarray]   # agent -> (n, m) weight on named fields
    p_err: dict[str, np.ndarray]      # agent -> (n,) probability of err (for baselines)
    covered: np.ndarray               # (n,) bool, rows with cached outputs


def excess_strength(p: float) -> float:
    return max(0.0, 2.0 * p - 1.0)


def load_llm_evidence(per_sample: Path, n: int, columns: list[str], strength=excess_strength) -> LLMEvidence:
    col_idx = {c.lower(): j for j, c in enumerate(columns)}
    m = len(columns)
    agents = ("auditor", "expert")
    row_pos = {a: np.zeros(n) for a in agents}
    row_neg = {a: np.zeros(n) for a in agents}
    cell_pos = {a: np.zeros((n, m)) for a in agents}
    p_err = {a: np.full(n, np.nan) for a in agents}
    covered = np.zeros(n, dtype=bool)
    with open(per_sample, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            i = int(r["row_id"])
            covered[i] = True
            for a in agents:
                out = r[a]
                p = float(out["p"])
                w = float(weight(strength(p)))
                if out["label"] == "err":
                    row_pos[a][i] = w
                    p_err[a][i] = p
                    for f in out.get("suspicious_fields", []) or []:
                        j = col_idx.get(str(f).strip().lower())
                        if j is not None:
                            cell_pos[a][i, j] = w
                else:
                    row_neg[a][i] = w
                    p_err[a][i] = 1.0 - p
    return LLMEvidence(row_pos, row_neg, cell_pos, p_err, covered)


# --------------------------------------------------------------------------
# Full fusion
# --------------------------------------------------------------------------
FAMILIES = ("syntax", "value", "llm")


def fuse(
    profiler_ev: dict[str, np.ndarray],
    family_of: dict[str, str],
    llm: LLMEvidence | None,
    rule: str = "family",
    use_llm: bool = True,
) -> dict[str, Masses]:
    """Fuse profiler and LLM evidence into cell- and row-level masses.

    rule = 'family'   : cautious (max) inside a family, Dempster (sum) across families
    rule = 'dempster' : Dempster (sum) over all sources (assumes all distinct)
    rule = 'cautious' : cautious (max) over all sources (assumes none distinct)
    """
    names = list(profiler_ev)
    n, m = next(iter(profiler_ev.values())).shape
    W = {k: weight(profiler_ev[k]) for k in names}

    def combine(ws: list[np.ndarray], fams: list[str]) -> np.ndarray:
        if not ws:
            return np.zeros_like(next(iter(W.values())))
        if rule == "dempster":
            return np.sum(ws, axis=0)
        if rule == "cautious":
            return np.max(ws, axis=0)
        out = 0.0
        for fam in set(fams):
            out = out + np.max([w for w, f in zip(ws, fams) if f == fam], axis=0)
        return out

    cell_ws = [W[k] for k in names]
    cell_fams = [family_of[k] for k in names]
    row_neg = np.zeros(n)
    row_llm_pos: list[np.ndarray] = []
    if use_llm and llm is not None:
        for a in llm.cell_pos:
            cell_ws.append(llm.cell_pos[a])
            cell_fams.append("llm")
            row_llm_pos.append(llm.row_pos[a])
        row_neg = np.max([llm.row_neg[a] for a in llm.row_neg], axis=0)
        if rule == "dempster":
            row_neg = np.sum([llm.row_neg[a] for a in llm.row_neg], axis=0)

    w_cell_pos = combine(cell_ws, cell_fams)
    cell = masses(w_cell_pos, np.repeat(row_neg[:, None], m, axis=1))

    # Row level: the strongest cell symptom of the data profilers (cautious over
    # cells), plus the LLM row verdicts as a separate family.
    prof_only = combine([W[k] for k in names], [family_of[k] for k in names])
    w_row_pos = prof_only.max(axis=1)
    if row_llm_pos:
        llm_row = np.sum(row_llm_pos, axis=0) if rule == "dempster" else np.max(row_llm_pos, axis=0)
        if rule == "cautious":
            w_row_pos = np.maximum(w_row_pos, llm_row)
        else:
            w_row_pos = w_row_pos + llm_row
    row = masses(w_row_pos, row_neg)
    return {"cell": cell, "row": row}


# --------------------------------------------------------------------------
# Review-budget calibration (label-free)
# --------------------------------------------------------------------------
def budget_threshold(cal_scores: np.ndarray, beta: float, rng: np.random.Generator) -> tuple[float, float]:
    """Threshold so that a fresh exchangeable row is sent to review with
    probability in [beta - 1/(n+1), beta].  Ties are broken by an independent
    uniform jitter, returned as (score threshold, jitter threshold)."""
    n = len(cal_scores)
    k = int(np.ceil((n + 1) * (1.0 - beta)))
    if k > n:
        return np.inf, np.inf
    u = rng.random(n)
    order = np.lexsort((u, cal_scores))  # ascending by (score, jitter)
    idx = order[k - 1]
    return float(cal_scores[idx]), float(u[idx])


def apply_budget(scores: np.ndarray, thr: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    u = rng.random(len(scores))
    s_thr, u_thr = thr
    return (scores > s_thr) | ((scores == s_thr) & (u > u_thr))
