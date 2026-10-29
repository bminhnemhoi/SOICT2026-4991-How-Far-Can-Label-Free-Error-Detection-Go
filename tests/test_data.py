"""Tests for src/data/{serialize,constraints,splits}.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.constraints import select_constraints
from src.data.serialize import compute_column_stats, serialize_row, serialize_row_with_constraints
from src.data.splits import make_splits


# --- serialize ---

def test_compute_column_stats_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "age": [30, 40, 50, None, 25],
            "state": ["CA", "CA", "NY", "CA", None],
        }
    )
    stats = compute_column_stats(df)
    assert stats["age"].dtype.startswith("float") or stats["age"].dtype.startswith("int")
    assert stats["age"].missing_rate == pytest.approx(0.2)
    assert stats["age"].min == 25.0
    assert stats["age"].max == 50.0
    assert stats["state"].missing_rate == pytest.approx(0.2)
    assert "CA" in stats["state"].top_values


def test_serialize_row_with_target_and_stats():
    df = pd.DataFrame({"age": [30, 40, 250], "name": ["A", "B", "C"]})
    stats = compute_column_stats(df)
    payload = serialize_row(
        df.iloc[2], row_id=2,
        target={"column": "age", "value": 250},
        column_stats=stats,
    )
    assert payload["row_id"] == 2
    assert payload["target"] == {"column": "age", "value": 250}
    assert payload["row"]["age"] == 250
    assert "age" in payload["column_stats"]
    assert "name" not in payload["column_stats"]  # target-only by default


def test_serialize_row_cleans_numpy_types():
    # Use dict input to avoid pandas upcasting int→float when mixed with NaN.
    row = {"a": np.int64(5), "b": np.float64(1.5), "c": np.nan, "d": np.bool_(True)}
    out = serialize_row(row)
    assert isinstance(out["row"]["a"], int) and out["row"]["a"] == 5
    assert isinstance(out["row"]["b"], float) and out["row"]["b"] == 1.5
    assert out["row"]["c"] is None
    assert isinstance(out["row"]["d"], bool) and out["row"]["d"] is True


# --- constraints ---

def test_select_constraints_by_overlap_and_severity():
    cs = [
        {"id": "C1", "severity": "high", "formal": {"lhs": ["ZIP"], "rhs": ["City"]}},
        {"id": "C2", "severity": "medium", "formal": {"column": "Phone"}},
        {"id": "C3", "severity": "high", "formal": {"column": "Age"}},
    ]
    sel = select_constraints(row_columns=["ZIP", "City", "Age"], all_constraints=cs, top_k=5)
    ids = [c["id"] for c in sel]
    assert "C1" in ids and "C3" in ids
    # C1 has overlap 2 (ZIP + City), C3 has overlap 1, both high severity
    assert ids.index("C1") < ids.index("C3")


def test_serialize_row_with_constraints_includes_nl():
    row = {"ZIP": "00000", "City": "?"}
    cs = [{"id": "C1", "nl": "ZIP→City", "severity": "high", "type": "fd",
           "formal": {"lhs": ["ZIP"], "rhs": ["City"]}}]
    out = serialize_row_with_constraints(row, cs)
    assert out["constraints"][0]["nl"] == "ZIP→City"
    assert out["constraints"][0]["id"] == "C1"


# --- splits ---

def test_make_splits_deterministic_and_disjoint():
    s1 = make_splits(100, train_frac=0.4, cal_frac=0.3, seed=42)
    s2 = make_splits(100, train_frac=0.4, cal_frac=0.3, seed=42)
    assert np.array_equal(s1.train, s2.train)
    assert np.array_equal(s1.cal, s2.cal)
    assert np.array_equal(s1.test, s2.test)

    all_ = np.concatenate([s1.train, s1.cal, s1.test])
    assert len(np.unique(all_)) == 100
    assert set(s1.train).isdisjoint(s1.cal)
    assert set(s1.cal).isdisjoint(s1.test)
    assert len(s1.train) == 40
    assert len(s1.cal) == 30
    assert len(s1.test) == 30


def test_make_splits_rejects_invalid_fracs():
    with pytest.raises(ValueError):
        make_splits(100, train_frac=0.7, cal_frac=0.4)
    with pytest.raises(ValueError):
        make_splits(0)
