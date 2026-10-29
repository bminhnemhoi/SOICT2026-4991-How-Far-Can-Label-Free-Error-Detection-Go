"""Row-level perturbation operators for aleatoric uncertainty (idea_main.md §4.5).

Each operator takes a DataFrame + target row index + params, returns the
perturbed row (leaving other rows and key columns unchanged by default).

Operators:
  - MissInject(p)      : set non-key cells to NULL with probability p
  - NumJitter(sigma)   : add N(0, sigma * col_std) noise to numeric non-key cells
  - CatSyn(k)          : replace categorical value with a random other value from same column
  - ContextShuffle(w)  : shuffle w neighbor rows in the context (affects ordering only)
  - EvidenceMask(q)    : mask q fraction of constraints (used by Expert prompt)

Constraints:
  - NEVER modify the cell under judgement (`protect_col`).
  - NEVER modify key/PK/FK columns passed via `protected_cols`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _non_protected_cols(df: pd.DataFrame, protected: Iterable[str]) -> list[str]:
    prot = set(protected)
    return [c for c in df.columns if c not in prot]


def miss_inject(
    row: pd.Series,
    *,
    p: float = 0.1,
    protected_cols: Iterable[str] = (),
    seed: int | None = None,
) -> pd.Series:
    """Set each non-protected cell to NaN with probability p."""
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"p must be in [0,1], got {p}")
    rng = _rng(seed)
    out = row.copy()
    prot = set(protected_cols)
    for col in out.index:
        if col in prot:
            continue
        if rng.random() < p:
            out[col] = np.nan
    return out


def num_jitter(
    row: pd.Series,
    df: pd.DataFrame,
    *,
    sigma: float = 0.01,
    protected_cols: Iterable[str] = (),
    seed: int | None = None,
) -> pd.Series:
    """Add Gaussian N(0, sigma * std_col) to numeric non-protected cells."""
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    rng = _rng(seed)
    out = row.copy()
    prot = set(protected_cols)
    for col in out.index:
        if col in prot:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        val = out[col]
        if pd.isna(val):
            continue
        std = float(df[col].dropna().astype(float).std(ddof=0))
        if std == 0 or np.isnan(std):
            continue
        out[col] = float(val) + rng.normal(0.0, sigma * std)
    return out


def cat_syn(
    row: pd.Series,
    df: pd.DataFrame,
    *,
    k: int = 1,
    protected_cols: Iterable[str] = (),
    seed: int | None = None,
) -> pd.Series:
    """For up to k categorical non-protected cells, replace value with another value from the same column's vocab."""
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    rng = _rng(seed)
    out = row.copy()
    prot = set(protected_cols)
    candidates = [
        c for c in out.index
        if c not in prot and not pd.api.types.is_numeric_dtype(df[c])
    ]
    rng.shuffle(candidates)
    changed = 0
    for col in candidates:
        if changed >= k:
            break
        cur = out[col]
        vocab = df[col].dropna().astype(str).unique().tolist()
        alt = [v for v in vocab if v != str(cur)]
        if not alt:
            continue
        out[col] = rng.choice(alt)
        changed += 1
    return out


def context_shuffle(
    neighbor_rows: Sequence[dict[str, Any]],
    *,
    w: int | None = None,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Shuffle the order of neighbor_rows (optionally the first w only)."""
    rng = _rng(seed)
    rows = list(neighbor_rows)
    if w is None or w >= len(rows):
        rng.shuffle(rows)
        return rows
    head = rows[:w]
    rng.shuffle(head)
    return head + rows[w:]


def evidence_mask(
    constraints: Sequence[dict[str, Any]],
    *,
    q: float = 0.2,
    protect_severity: str | None = "high",
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Randomly drop fraction q of constraints from the list.

    By default never drops `severity == protect_severity` (so 'high' severity is preserved).
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"q must be in [0,1], got {q}")
    rng = _rng(seed)
    kept: list[dict[str, Any]] = []
    for c in constraints:
        if protect_severity is not None and c.get("severity") == protect_severity:
            kept.append(c)
            continue
        if rng.random() >= q:
            kept.append(c)
    return kept


@dataclass(frozen=True)
class PerturbSpec:
    """Config for a perturbation family (applied per-call to generate U_ale samples)."""

    op: str  # 'miss_inject' | 'num_jitter' | 'cat_syn' | 'context_shuffle' | 'evidence_mask'
    params: dict[str, Any]


def apply_perturbation(
    spec: PerturbSpec,
    row: pd.Series,
    df: pd.DataFrame,
    *,
    protected_cols: Iterable[str] = (),
    seed: int | None = None,
) -> pd.Series:
    """Dispatch row-level perturbation by spec.op. (context_shuffle/evidence_mask act on other inputs.)"""
    if spec.op == "miss_inject":
        return miss_inject(row, protected_cols=protected_cols, seed=seed, **spec.params)
    if spec.op == "num_jitter":
        return num_jitter(row, df, protected_cols=protected_cols, seed=seed, **spec.params)
    if spec.op == "cat_syn":
        return cat_syn(row, df, protected_cols=protected_cols, seed=seed, **spec.params)
    raise ValueError(f"Unknown or non-row perturbation op: {spec.op!r}")
