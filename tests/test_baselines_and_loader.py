"""Tests for baselines and RAHA loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from src.baselines import BASELINES, debate_no_ds, majority_vote, single_llm, bcl_ds_only
from src.bcl.bcl import BCLWeights
from src.data.loaders.raha import CANONICAL_RENAME, load_raha_dataset


@dataclass
class _Out:
    label: str
    p: float
    c: float = 1.0
    a: float = 1.0


def test_single_llm_commits_on_err():
    a = _Out("err", 0.9)
    e = _Out("clean", 0.9)  # ignored
    r = single_llm(a, e)
    assert r["decision"] == "error"
    assert r["p_hat"] == pytest.approx(0.9)


def test_single_llm_abstains_low_conf():
    r = single_llm(_Out("err", 0.3), _Out("clean", 0.5))
    assert r["decision"] == "abstain"


def test_majority_vote_agree():
    r = majority_vote(_Out("err", 0.8), _Out("err", 0.7))
    assert r["decision"] == "error"
    assert r["p_hat"] == pytest.approx(0.75)


def test_majority_vote_disagree_abstains():
    r = majority_vote(_Out("err", 0.8), _Out("clean", 0.7))
    assert r["decision"] == "abstain"


def test_debate_no_ds_err_above_thresh():
    r = debate_no_ds(_Out("err", 0.9), _Out("err", 0.7), w_a=0.5, w_b=0.5)
    assert r["decision"] == "error"
    assert r["p_hat"] >= 0.6


def test_debate_no_ds_abstain_middle():
    r = debate_no_ds(_Out("err", 0.5), _Out("clean", 0.5), w_a=0.5, w_b=0.5)
    assert r["decision"] == "abstain"


def test_bcl_ds_only_returns_valid_p_hat():
    r = bcl_ds_only(_Out("err", 0.8), _Out("err", 0.7), weights=BCLWeights())
    assert 0.0 <= r["p_hat"] <= 1.0
    assert r["decision"] in ("error", "clean", "abstain")


def test_BASELINES_registry():
    assert set(BASELINES) == {"single_llm", "majority_vote", "debate_no_ds", "bcl_ds_only"}


# ---- RAHA loader -----------------------------------------------------------


def _make_tmp_dataset(tmp_path: Path) -> Path:
    ds = tmp_path / "mini"
    ds.mkdir()
    dirty = pd.DataFrame([
        {"id": 1, "provider_number": "10018", "zip": "99999"},   # zip differs
        {"id": 2, "provider_number": "10019", "zip": "35233"},   # clean
        {"id": 3, "provider_number": "10020", "zip": "35234"},   # clean
    ])
    clean = pd.DataFrame([
        {"id": 1, "ProviderNumber": "10018", "ZIP": "35233"},
        {"id": 2, "ProviderNumber": "10019", "ZIP": "35233"},
        {"id": 3, "ProviderNumber": "10020", "ZIP": "35234"},
    ])
    dirty.to_csv(ds / "dirty.csv", index=False)
    clean.to_csv(ds / "clean.csv", index=False)
    return ds


def test_load_raha_detects_diff(tmp_path):
    ds_dir = _make_tmp_dataset(tmp_path)
    ds = load_raha_dataset(ds_dir.parent, ds_dir.name)
    assert ds.n_rows == 3
    # Row 0 has zip diff, rows 1-2 are clean.
    assert ds.dirty_cells[0] == ["zip"]
    assert ds.dirty_cells[1] == []
    assert ds.dirty_cells[2] == []


def test_to_maled_frame_with_rename(tmp_path):
    ds_dir = _make_tmp_dataset(tmp_path)
    ds = load_raha_dataset(ds_dir.parent, ds_dir.name)
    df = ds.to_maled_frame(rename={"zip": "ZIP", "provider_number": "ProviderNumber"})
    assert "ZIP" in df.columns
    assert df.loc[0, "y_true"] == "err"
    assert df.loc[1, "y_true"] == "clean"
    assert df.loc[0, "err_type"] == "pattern_violation"


def test_canonical_rename_has_hospital():
    assert "hospital" in CANONICAL_RENAME
    assert CANONICAL_RENAME["hospital"]["zip"] == "ZIP"
