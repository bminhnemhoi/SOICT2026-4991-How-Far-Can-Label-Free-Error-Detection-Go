"""Row serialization + column statistics for agent prompts.

Auditor needs: row + column_stats (mean/std/min/max/missing_rate/dtype).
Expert needs: row + selected constraints (NL form).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ColumnStats:
    dtype: str
    missing_rate: float
    n_unique: int
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    top_values: list[Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"dtype": self.dtype, "missing_rate": self.missing_rate, "n_unique": self.n_unique}
        for k in ("mean", "std", "min", "max", "top_values"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        return d


def compute_column_stats(df: pd.DataFrame, top_k: int = 5) -> dict[str, ColumnStats]:
    """Compute column-level stats used by the Auditor prompt."""
    out: dict[str, ColumnStats] = {}
    n = len(df)
    for col in df.columns:
        s = df[col]
        missing_rate = float(s.isna().mean()) if n else 0.0
        n_unique = int(s.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(s) and s.notna().any():
            s_num = s.dropna().astype(float)
            out[col] = ColumnStats(
                dtype=str(s.dtype),
                missing_rate=missing_rate,
                n_unique=n_unique,
                mean=float(s_num.mean()),
                std=float(s_num.std(ddof=0)) if len(s_num) > 0 else 0.0,
                min=float(s_num.min()),
                max=float(s_num.max()),
            )
        else:
            top = s.dropna().astype(str).value_counts().head(top_k).index.tolist()
            out[col] = ColumnStats(
                dtype=str(s.dtype), missing_rate=missing_rate, n_unique=n_unique, top_values=top
            )
    return out


def _clean_value(v: Any) -> Any:
    """Convert numpy/pandas scalars to JSON-serializable native types."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def serialize_row(
    row: pd.Series | dict[str, Any],
    *,
    row_id: int | None = None,
    target: dict[str, Any] | None = None,
    column_stats: dict[str, ColumnStats] | None = None,
    include_stats_for_target_only: bool = True,
) -> dict[str, Any]:
    """Pack a row into the JSON payload expected by the Auditor prompt.

    Args:
        row: a pandas Series or dict {col: value}.
        row_id: int id to include.
        target: optional {column, value} to focus judgement.
        column_stats: precomputed column stats. If None, field omitted.
        include_stats_for_target_only: if True and target set, only emit stats for
            that column (keeps prompt small).
    """
    if isinstance(row, pd.Series):
        row_dict = {str(k): _clean_value(v) for k, v in row.to_dict().items()}
    else:
        row_dict = {str(k): _clean_value(v) for k, v in row.items()}

    payload: dict[str, Any] = {"row": row_dict}
    if row_id is not None:
        payload["row_id"] = int(row_id)
    if target is not None:
        payload["target"] = {"column": str(target["column"]), "value": _clean_value(target.get("value"))}
    if column_stats:
        if include_stats_for_target_only and target is not None:
            col = str(target["column"])
            if col in column_stats:
                payload["column_stats"] = {col: column_stats[col].as_dict()}
        else:
            payload["column_stats"] = {c: s.as_dict() for c, s in column_stats.items()}
    return payload


def serialize_row_with_constraints(
    row: pd.Series | dict[str, Any],
    constraints: Iterable[dict[str, Any]],
    *,
    row_id: int | None = None,
    target: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pack row + selected NL constraints for the Expert prompt."""
    base = serialize_row(row, row_id=row_id, target=target)
    base["constraints"] = [
        {"id": c.get("id"), "nl": c.get("nl"), "severity": c.get("severity"),
         "type": c.get("type"), "error_type_when_violated": c.get("error_type_when_violated")}
        for c in constraints
    ]
    return base
