"""Tests for src/uq/perturb.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.uq.perturb import (
    PerturbSpec,
    apply_perturbation,
    cat_syn,
    context_shuffle,
    evidence_mask,
    miss_inject,
    num_jitter,
)


@pytest.fixture
def df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],                   # PK
            "age": [30.0, 40.0, 50.0, 45.0, 35.0],
            "city": ["NYC", "NYC", "Boston", "Boston", "LA"],
            "state": ["NY", "NY", "MA", "MA", "CA"],
        }
    )


def test_miss_inject_respects_protected(df):
    row = df.iloc[0]
    out = miss_inject(row, p=1.0, protected_cols=["id", "age"], seed=0)
    assert out["id"] == row["id"]           # protected
    assert out["age"] == row["age"]         # protected
    assert pd.isna(out["city"])             # nulled
    assert pd.isna(out["state"])


def test_miss_inject_p0_is_identity(df):
    row = df.iloc[0]
    out = miss_inject(row, p=0.0, seed=0)
    assert out.equals(row)


def test_num_jitter_only_touches_numeric_non_protected(df):
    row = df.iloc[0]
    out = num_jitter(row, df, sigma=0.1, protected_cols=["id"], seed=0)
    assert out["id"] == row["id"]
    assert out["city"] == row["city"]       # categorical untouched
    assert out["age"] != row["age"]         # numeric perturbed


def test_num_jitter_zero_sigma_is_identity(df):
    row = df.iloc[0]
    out = num_jitter(row, df, sigma=0.0, seed=0)
    assert out["age"] == row["age"]


def test_cat_syn_replaces_k_categoricals(df):
    row = df.iloc[0]  # {city: NYC, state: NY}
    out = cat_syn(row, df, k=2, protected_cols=["id"], seed=42)
    changed = sum(1 for c in ("city", "state") if out[c] != row[c])
    assert changed >= 1  # at least one changed (k=2 may randomize order)


def test_cat_syn_k_zero_is_identity(df):
    row = df.iloc[0]
    out = cat_syn(row, df, k=0, seed=0)
    assert out.equals(row)


def test_context_shuffle_preserves_content():
    rows = [{"id": i} for i in range(5)]
    out = context_shuffle(rows, seed=1)
    assert sorted(r["id"] for r in out) == [0, 1, 2, 3, 4]
    assert len(out) == 5


def test_evidence_mask_protects_high_severity():
    cs = [
        {"id": "A", "severity": "high"},
        {"id": "B", "severity": "medium"},
        {"id": "C", "severity": "low"},
        {"id": "D", "severity": "medium"},
    ]
    out = evidence_mask(cs, q=1.0, protect_severity="high", seed=0)
    ids = {c["id"] for c in out}
    assert "A" in ids               # high preserved
    assert "B" not in ids and "C" not in ids and "D" not in ids  # all dropped


def test_evidence_mask_q0_is_identity():
    cs = [{"id": "X", "severity": "low"}]
    assert evidence_mask(cs, q=0.0, seed=0) == cs


def test_apply_perturbation_dispatch(df):
    row = df.iloc[0]
    spec = PerturbSpec(op="num_jitter", params={"sigma": 0.05})
    out = apply_perturbation(spec, row, df, protected_cols=["id"], seed=7)
    assert out["id"] == row["id"]
    assert out["age"] != row["age"]

    with pytest.raises(ValueError):
        apply_perturbation(PerturbSpec(op="unknown", params={}), row, df)
