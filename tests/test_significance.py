"""Sanity tests for bootstrap CI + paired permutation test."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.significance import bootstrap_ci, paired_permutation_test


def test_bootstrap_ci_perfect_predictions() -> None:
    """Perfect F1 → CI collapses around 1.0."""
    yt = np.array([1] * 50 + [0] * 50)
    yp = yt.copy()
    ps = yt.astype(float)
    out = bootstrap_ci(yt, yp, ps, metric="f1", n_boot=500, seed=0)
    assert out["point"] == pytest.approx(1.0)
    assert out["ci_lo"] == pytest.approx(1.0)
    assert out["ci_hi"] == pytest.approx(1.0)


def test_bootstrap_ci_contains_point() -> None:
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 2, 200)
    yp = rng.integers(0, 2, 200)
    ps = rng.random(200)
    out = bootstrap_ci(yt, yp, ps, metric="f1", n_boot=500, seed=0)
    assert out["ci_lo"] <= out["point"] <= out["ci_hi"]


def test_permutation_identical_runs_pvalue_one() -> None:
    rng = np.random.default_rng(0)
    yt = rng.integers(0, 2, 100)
    yp = rng.integers(0, 2, 100)
    ps = rng.random(100)
    out = paired_permutation_test(yt, yp, ps, yt, yp, ps, metric="f1", n_perm=200, seed=1)
    assert out["diff_b_minus_a"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)


def test_permutation_huge_difference_significant() -> None:
    """B perfect, A all wrong → p should be very small."""
    yt = np.array([1] * 50 + [0] * 50)
    yp_a = 1 - yt   # always wrong
    yp_b = yt.copy()  # always right
    ps_a = (1 - yt).astype(float)
    ps_b = yt.astype(float)
    out = paired_permutation_test(yt, yp_a, ps_a, yt, yp_b, ps_b, metric="f1", n_perm=500, seed=0)
    assert out["diff_b_minus_a"] > 0.5
    assert out["p_value"] < 0.05


def test_permutation_requires_aligned_y_true() -> None:
    yt_a = np.array([1, 0, 1, 0])
    yt_b = np.array([0, 0, 1, 0])
    yp = np.array([0, 0, 0, 0])
    ps = yp.astype(float)
    with pytest.raises(ValueError):
        paired_permutation_test(yt_a, yp, ps, yt_b, yp, ps, metric="f1", n_perm=10)
