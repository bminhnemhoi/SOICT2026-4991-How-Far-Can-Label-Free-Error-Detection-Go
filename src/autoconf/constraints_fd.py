"""L2b - Approximate functional-dependency mining (noise-robust).

For every candidate pair (LHS=X, RHS=Y) we compute a *majority confidence*

    conf_maj(X -> Y) = (1/n) * sum over LHS-groups g of |{r in g : Y(r) = mode(Y|g)}|

i.e., the fraction of rows that match the modal RHS within their LHS group.
On a dirty dataset where ~e fraction of rows violate the FD, conf_maj ~= 1 - e/2
even for a true FD, whereas the strict "all-unique" confidence collapses fast.

LHS size is bounded by max_lhs (default 2) to stay tractable; the number of
candidates is O(C(n_cols, k) * n_cols).
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

from src.autoconf.profiler import DatasetProfile  # noqa: F401  (api)


def mine_fds(
    df: pd.DataFrame,
    *,
    max_lhs: int = 2,
    min_confidence: float = 0.95,
    min_group_support: int = 2,
    min_distinct_groups: int = 5,
    exclude_cols: list[str] | None = None,
    skip_constant_rhs: bool = True,
    skip_key_lhs: bool = True,
    key_uniqueness_threshold: float = 0.98,
    collapse_equivalence: bool = False,
    equivalence_min_bidir_conf: float = 0.99,
) -> list[dict[str, Any]]:
    """Mine approximate FDs X -> Y from a (possibly dirty) DataFrame.

    Args:
        df: input dataframe.
        max_lhs: maximum number of columns on the LHS (1 or 2 recommended).
        min_confidence: keep FDs whose row-weighted confidence >= this.
        min_group_support: skip FDs where most groups have fewer than this many rows
            (unstable).  We require median(group size) >= min_group_support.
        min_distinct_groups: skip FDs with fewer distinct LHS values than this.
        exclude_cols: columns to skip entirely.
        skip_constant_rhs: drop FDs whose RHS column has <=1 distinct value
            (trivial: any LHS implies a constant).
        skip_key_lhs: drop FDs whose LHS has near-row-count uniqueness
            (trivial: a unique-id column implies everything).
        key_uniqueness_threshold: LHS uniqueness ratio above which we skip
            (default 0.95 -> n_unique >= 0.95*n is treated as a key).
    """
    cols = [c for c in df.columns if not exclude_cols or c not in exclude_cols]
    n = len(df)
    if n == 0 or len(cols) < 2:
        return []

    nunique_per_col = {c: int(df[c].nunique(dropna=False)) for c in cols}
    constant_cols = {c for c, u in nunique_per_col.items() if u <= 1}
    key_cols = (
        {c for c, u in nunique_per_col.items() if u >= key_uniqueness_threshold * n}
        if skip_key_lhs else set()
    )

    out: list[dict[str, Any]] = []

    for k in range(1, max_lhs + 1):
        for lhs in combinations(cols, k):
            lhs_list = list(lhs)
            if key_cols and any(c in key_cols for c in lhs_list):
                continue
            try:
                grp = df.groupby(lhs_list, dropna=False, observed=True)
            except Exception:
                continue
            n_groups = grp.ngroups
            if n_groups < min_distinct_groups:
                continue
            sizes = grp.size()
            if float(sizes.median()) < float(min_group_support):
                continue

            for rhs in cols:
                if rhs in lhs:
                    continue
                if skip_constant_rhs and rhs in constant_cols:
                    continue
                # Majority confidence: rows matching the modal RHS within their group.
                try:
                    modal_match = grp[rhs].transform(
                        lambda s: (s == s.mode(dropna=False).iloc[0]).astype(int)
                        if len(s.mode(dropna=False)) else 0
                    )
                except Exception:
                    continue
                conf = float(modal_match.sum()) / n
                if conf >= min_confidence:
                    out.append(_mk_fd(lhs_list, rhs, conf, n_groups))

    out = _prune_non_minimal(out)
    if collapse_equivalence:
        out = collapse_equivalent_lhs(out, min_bidir_conf=equivalence_min_bidir_conf)
    return out


def _mk_fd(lhs: list[str], rhs: str, conf: float, n_groups: int) -> dict[str, Any]:
    lhs_str = "+".join(lhs)
    sev = "high" if conf > 0.99 else "medium"
    return {
        "id": f"AUTO-FD-{lhs_str}->{rhs}",
        "type": "functional_dependency",
        "nl": f"{' + '.join(lhs)} determines {rhs} (soft-confidence {conf:.3f}).",
        "formal": {"lhs": list(lhs), "rhs": [rhs]},
        "severity": sev,
        "error_type_when_violated": "fd_violation",
        "source": "auto_fd",
        "confidence": round(conf, 4),
        "n_groups": int(n_groups),
    }


def _prune_non_minimal(fds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove FDs (A, B) -> Y when A -> Y already holds with higher/equal conf."""
    by_rhs: dict[str, list[dict[str, Any]]] = {}
    for f in fds:
        rhs = f["formal"]["rhs"][0]
        by_rhs.setdefault(rhs, []).append(f)

    keep: list[dict[str, Any]] = []
    for rhs, group in by_rhs.items():
        group_sorted = sorted(group, key=lambda f: (len(f["formal"]["lhs"]), -f["confidence"]))
        minimal: list[dict[str, Any]] = []
        for f in group_sorted:
            lhs_set = set(f["formal"]["lhs"])
            dominated = False
            for m in minimal:
                m_lhs = set(m["formal"]["lhs"])
                if m_lhs.issubset(lhs_set) and m["confidence"] >= f["confidence"] - 1e-9:
                    dominated = True
                    break
            if not dominated:
                minimal.append(f)
        keep.extend(minimal)
    return keep


def _equivalence_classes(fds: list[dict[str, Any]], min_bidir_conf: float = 0.95) -> dict[str, str]:
    """Cluster columns where A -> B and B -> A both hold with high confidence.

    Returns a mapping col -> representative_col. Used to dedupe FDs that are
    transitive consequences of equivalence (e.g., brewery_id <-> brewery_name
    means brewery_id -> X is redundant whenever brewery_name -> X is kept).
    """
    by_pair: dict[tuple[str, str], float] = {}
    for f in fds:
        lhs = f["formal"]["lhs"]
        rhs = f["formal"]["rhs"]
        if len(lhs) == 1 and len(rhs) == 1:
            by_pair[(lhs[0], rhs[0])] = float(f.get("confidence", 0.0))

    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent.get(x, x), parent.get(x, x))
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    cols = {c for pair in by_pair for c in pair}
    for c in cols:
        parent[c] = c

    for (a, b), conf_ab in by_pair.items():
        conf_ba = by_pair.get((b, a), 0.0)
        if conf_ab >= min_bidir_conf and conf_ba >= min_bidir_conf:
            union(a, b)

    return {c: find(c) for c in cols}


def collapse_equivalent_lhs(fds: list[dict[str, Any]], min_bidir_conf: float = 0.95) -> list[dict[str, Any]]:
    """Drop FDs whose LHS is equivalent (bidirectional FD) to a kept LHS for the same RHS.

    Keeps one FD per (equivalence-class-of-LHS, RHS) pair, preferring smallest LHS
    then highest confidence.
    """
    eq = _equivalence_classes(fds, min_bidir_conf=min_bidir_conf)
    seen: dict[tuple[frozenset, str], dict[str, Any]] = {}
    for f in sorted(fds, key=lambda x: (len(x["formal"]["lhs"]), -float(x.get("confidence", 0.0)))):
        rhs = f["formal"]["rhs"][0]
        rep_lhs = frozenset(eq.get(c, c) for c in f["formal"]["lhs"])
        rep_rhs = eq.get(rhs, rhs)
        # do not collapse FDs where LHS-rep == RHS-rep (would be self-implying)
        if rep_rhs in rep_lhs:
            continue
        key = (rep_lhs, rep_rhs)
        if key not in seen:
            seen[key] = f
    return list(seen.values())
