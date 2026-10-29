"""Constraint loader + relevance-based selection.

Constraint YAML format: see docs/04-constraints-catalog.md.
Each constraint: {id, type, nl, formal, severity, error_type_when_violated}.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from src.utils.io import read_yaml

_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def load_constraints(path: str | Path) -> list[dict[str, Any]]:
    """Load a constraint YAML file. Returns the list under `constraints` key."""
    doc = read_yaml(path)
    cs = doc.get("constraints", [])
    if not isinstance(cs, list):
        raise ValueError(f"{path}: 'constraints' must be a list, got {type(cs).__name__}")
    return cs


def _columns_in_constraint(c: dict[str, Any]) -> set[str]:
    """Extract all columns referenced by a constraint's formal spec."""
    formal = c.get("formal") or {}
    cols: set[str] = set()
    for key in ("lhs", "rhs"):
        v = formal.get(key)
        if isinstance(v, list):
            cols.update(str(x) for x in v)
    col = formal.get("column")
    if isinstance(col, str):
        cols.add(col)
    elif isinstance(col, list):
        cols.update(str(x) for x in col)
    # denial / temporal: try best-effort extraction
    same = (formal.get("forbid_pattern") or {}).get("same", [])
    if isinstance(same, list):
        cols.update(str(x) for x in same)
    return cols


def select_constraints(
    row_columns: Iterable[str],
    all_constraints: list[dict[str, Any]],
    *,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Return constraints whose columns intersect row_columns, sorted by severity.

    If fewer than top_k relevant, fall back to all constraints (truncated).
    """
    cols = {str(c) for c in row_columns}
    scored: list[tuple[int, int, dict[str, Any]]] = []
    for c in all_constraints:
        c_cols = _columns_in_constraint(c)
        overlap = len(c_cols & cols)
        sev_rank = _SEVERITY_ORDER.get(c.get("severity", "medium"), 1)
        if overlap > 0:
            scored.append((-overlap, sev_rank, c))  # more overlap first
    scored.sort(key=lambda t: (t[0], t[1]))
    relevant = [c for _, _, c in scored]
    if len(relevant) < top_k:
        # pad with high-severity constraints not already included
        seen_ids = {c.get("id") for c in relevant}
        extra = sorted(
            (c for c in all_constraints if c.get("id") not in seen_ids),
            key=lambda c: _SEVERITY_ORDER.get(c.get("severity", "medium"), 1),
        )
        relevant.extend(extra[: top_k - len(relevant)])
    return relevant[:top_k]
