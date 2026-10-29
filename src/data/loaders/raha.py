"""Loader for RAHA-style dataset directories (dirty.csv + clean.csv).

Row-level label: y_true = 'err' if ANY cell differs from clean, else 'clean'.
Cell-level ground truth is also extracted (list of dirty column names per row).

Supported dirty-column renames (dirty uses snake_case; clean uses PascalCase).
A column alignment dict is built automatically by normalizing both sides.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

_SNAKE = re.compile(r"(?<!^)(?=[A-Z])")


# Canonical column names (PascalCase) matching data/constraints/<name>.yaml.
CANONICAL_RENAME: dict[str, dict[str, str]] = {
    "hospital": {
        "provider_number": "ProviderNumber",
        "name": "HospitalName",
        "address_1": "Address",
        "address_2": "Address2",
        "address_3": "Address3",
        "city": "City",
        "state": "State",
        "zip": "ZIP",
        "county": "CountyName",
        "phone": "PhoneNumber",
        "type": "HospitalType",
        "owner": "HospitalOwner",
        "emergency_service": "EmergencyService",
        "condition": "Condition",
        "measure_code": "MeasureCode",
        "measure_name": "MeasureName",
        "score": "Score",
        "sample": "Sample",
        "state_average": "Stateavg",
    },
    "flights": {
        # RAHA flights columns are already Capitalized; keep identity for most.
    },
    "beers": {
        # RAHA beers columns already aligned; keep identity.
    },
}


def _normalize_name(s: str) -> str:
    """Lowercase + strip non-alnum, so 'ProviderNumber' == 'provider_number'."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


@dataclass
class RahaDataset:
    name: str
    dirty: pd.DataFrame
    clean: pd.DataFrame
    dirty_cells: list[list[str]]  # per-row list of dirty column names

    @property
    def n_rows(self) -> int:
        return len(self.dirty)

    def to_maled_frame(self, keep_cols: Iterable[str] | None = None, rename: dict[str, str] | None = None) -> pd.DataFrame:
        """Produce a DataFrame compatible with the MA-LED runner.

        Columns: row_id, <data cols...>, y_true ('err'|'clean'), err_type (str|None).
        `rename` (optional) maps raw dirty-col names -> canonical names
        (used to align with constraints YAMLs). `dirty_cells` are also remapped.
        """
        df = self.dirty.copy()
        if rename:
            df = df.rename(columns=rename)
        if keep_cols:
            keep = [c for c in keep_cols if c in df.columns]
            df = df[keep]
        df.insert(0, "row_id", range(len(df)))
        # Remap dirty_cells col names through `rename` (if any).
        rmap = rename or {}
        remapped_cells = [[rmap.get(c, c) for c in cells] for cells in self.dirty_cells]
        y_true = ["err" if cells else "clean" for cells in remapped_cells]
        err_type = [_infer_err_type(cells) if cells else "none" for cells in remapped_cells]
        df["y_true"] = y_true
        df["err_type"] = err_type
        return df.reset_index(drop=True)


def _infer_err_type(dirty_cells: list[str]) -> str:
    """Heuristic err-type assignment from column names of dirty cells.

    Conservative: always return 'typo_or_pattern' as RAHA datasets are typo-heavy.
    Subclasses may override.
    """
    joined = " ".join(c.lower() for c in dirty_cells)
    if "zip" in joined or "phone" in joined or "code" in joined:
        return "pattern_violation"
    if "name" in joined or "address" in joined or "city" in joined:
        return "typo"
    return "typo"


def load_raha_dataset(root: str | Path, name: str | None = None) -> RahaDataset:
    """Load a RAHA-style dataset from <root>/<name>/{dirty,clean}.csv.

    If `name` is None, `root` is treated as the dataset dir itself.
    """
    root = Path(root)
    ds_dir = root / name if name else root
    dirty_path = ds_dir / "dirty.csv"
    clean_path = ds_dir / "clean.csv"
    if not dirty_path.exists() or not clean_path.exists():
        raise FileNotFoundError(f"Expected {dirty_path} and {clean_path}")

    dirty = pd.read_csv(dirty_path, dtype=str, keep_default_na=False)
    clean = pd.read_csv(clean_path, dtype=str, keep_default_na=False)

    if len(dirty) != len(clean):
        raise ValueError(f"dirty/clean row count mismatch: {len(dirty)} vs {len(clean)}")

    # Align columns by normalized name.
    dirty_map = {_normalize_name(c): c for c in dirty.columns}
    clean_map = {_normalize_name(c): c for c in clean.columns}
    common = [k for k in dirty_map if k in clean_map]
    if not common:
        raise ValueError("No columns align between dirty and clean.")

    # Compute per-row list of dirty-column names (using dirty's column labels).
    dirty_cells: list[list[str]] = []
    for i in range(len(dirty)):
        diffs: list[str] = []
        for key in common:
            dcol, ccol = dirty_map[key], clean_map[key]
            if dirty.at[i, dcol] != clean.at[i, ccol]:
                diffs.append(dcol)
        dirty_cells.append(diffs)

    return RahaDataset(
        name=(name or ds_dir.name),
        dirty=dirty.reset_index(drop=True),
        clean=clean.reset_index(drop=True),
        dirty_cells=dirty_cells,
    )
