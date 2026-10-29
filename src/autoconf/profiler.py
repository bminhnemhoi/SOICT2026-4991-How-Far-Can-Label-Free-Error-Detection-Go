"""L1 — Schema profiler for arbitrary tabular data.

Produces a DatasetProfile (schema + column stats + sample rows) that the
downstream constraint miners and prompt templater consume.  No LLM calls,
deterministic, cheap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.serialize import compute_column_stats


@dataclass(frozen=True)
class DatasetProfile:
    """Structured summary of a (possibly dirty) tabular dataset."""

    name: str
    n_rows: int
    n_cols: int
    columns: list[str]
    column_stats: dict[str, dict[str, Any]]
    sample_rows: list[dict[str, Any]]
    inferred_domain: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": self.columns,
            "column_stats": self.column_stats,
            "sample_rows": self.sample_rows,
            "inferred_domain": self.inferred_domain,
            "meta": dict(self.meta),
        }


def _clean(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def profile_dataframe(
    df: pd.DataFrame,
    *,
    name: str = "unnamed",
    sample_k: int = 20,
    seed: int = 42,
    exclude_cols: list[str] | None = None,
) -> DatasetProfile:
    """Build a DatasetProfile from a DataFrame."""
    cols = list(df.columns)
    if exclude_cols:
        cols = [c for c in cols if c not in exclude_cols]
        df = df[cols].copy()

    stats = compute_column_stats(df)
    stats_dict = {c: s.as_dict() for c, s in stats.items()}

    k = min(sample_k, len(df))
    sample_df = df.sample(n=k, random_state=seed) if k > 0 else df.head(0)
    sample_rows = [
        {str(c): _clean(v) for c, v in row.items()}
        for row in sample_df.to_dict(orient="records")
    ]

    return DatasetProfile(
        name=name,
        n_rows=len(df),
        n_cols=len(cols),
        columns=cols,
        column_stats=stats_dict,
        sample_rows=sample_rows,
    )


def profile_csv(path: str | Path, **kwargs: Any) -> DatasetProfile:
    df = pd.read_csv(path)
    name = kwargs.pop("name", Path(path).stem)
    return profile_dataframe(df, name=name, **kwargs)


def profile_parquet(path: str | Path, **kwargs: Any) -> DatasetProfile:
    df = pd.read_parquet(path)
    name = kwargs.pop("name", Path(path).stem)
    return profile_dataframe(df, name=name, **kwargs)
