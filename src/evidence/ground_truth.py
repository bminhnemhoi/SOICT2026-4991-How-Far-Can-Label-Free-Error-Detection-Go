"""Cell-level ground truth for the RAHA benchmarks with POSITIONAL column alignment.

The earlier loader (src/data/loaders/raha.py) aligned dirty/clean columns by
normalised name.  On Hospital, 7 of 19 data columns have different names in
dirty.csv and clean.csv (e.g. ``name`` vs ``HospitalName``, ``zip`` vs
``ZipCode``), so errors in those columns were never counted and the row-level
labels under-reported the error prevalence (0.268 instead of 0.407).  RAHA's
own evaluation compares the two files positionally; we do the same here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "raha" / "datasets"
PROC = ROOT / "data" / "processed"

# Identifier / bookkeeping columns that are not part of the data being judged.
ID_COLS = {"row_id", "index", "tuple_id", "id", "y_true", "err_type"}


@dataclass
class Benchmark:
    name: str
    dirty: pd.DataFrame          # n x m data columns, canonical names, str values
    errors: np.ndarray           # n x m bool, True where dirty != clean
    columns: list[str]
    splits: dict[str, np.ndarray]  # split name -> row ids (positions)
    raw_col_index: list[int]       # position of each data column in dirty.csv

    @property
    def row_err(self) -> np.ndarray:
        return self.errors.any(axis=1)


def load_benchmark(name: str) -> Benchmark:
    dirty_raw = pd.read_csv(RAW / name / "dirty.csv", dtype=str, keep_default_na=False)
    clean_raw = pd.read_csv(RAW / name / "clean.csv", dtype=str, keep_default_na=False)
    if dirty_raw.shape != clean_raw.shape:
        raise ValueError(f"{name}: dirty/clean shape mismatch {dirty_raw.shape} vs {clean_raw.shape}")

    allp = pd.read_parquet(PROC / f"{name}_v1" / "all.parquet")
    assert (allp["row_id"].to_numpy() == np.arange(len(allp))).all()
    canon = [c for c in allp.columns if c not in ID_COLS]

    # Map canonical (processed) names onto raw dirty positions.  The processed
    # frame keeps the raw column order, so we align by position after removing
    # identifier columns on both sides.
    raw_data_idx = [i for i, c in enumerate(dirty_raw.columns) if c not in ID_COLS]
    if len(raw_data_idx) != len(canon):
        raise ValueError(f"{name}: column count mismatch {len(raw_data_idx)} vs {len(canon)}")

    dirty = dirty_raw.iloc[:, raw_data_idx].copy()
    clean = clean_raw.iloc[:, raw_data_idx].copy()
    dirty.columns = canon
    clean.columns = canon
    # Sanity: processed values must equal raw dirty values.
    same = (allp[canon].astype(str).to_numpy() == dirty.to_numpy()).mean()
    if same < 0.999:
        raise ValueError(f"{name}: processed/raw value agreement only {same:.4f}")

    errors = dirty.to_numpy() != clean.to_numpy()
    splits = {
        s: np.sort(pd.read_parquet(PROC / f"{name}_v1" / f"{s}.parquet")["row_id"].to_numpy().astype(int))
        for s in ("test", "dev", "cal")
    }
    return Benchmark(name=name, dirty=dirty.reset_index(drop=True), errors=errors, columns=canon,
                     splits=splits, raw_col_index=raw_data_idx)


if __name__ == "__main__":
    for ds in ("hospital", "flights", "beers", "rayyan"):
        b = load_benchmark(ds)
        t = b.splits["test"]
        print(f"{ds:9s} n={len(b.dirty):5d} m={len(b.columns):2d} cell_err={b.errors.mean():.3f} "
              f"row_prev(all)={b.row_err.mean():.3f} row_prev(test)={b.row_err[t].mean():.3f} "
              f"n_test={len(t)} n_err_cells_test={int(b.errors[t].sum())}")
