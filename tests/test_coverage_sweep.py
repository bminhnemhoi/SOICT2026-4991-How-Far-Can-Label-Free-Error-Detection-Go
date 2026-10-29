"""Tests for coverage_sweep — validity of conformal abstention rate."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.coverage_sweep import _conformal_quantile, sweep_one


def test_conformal_quantile_finite_sample_correction() -> None:
    # n=10, α=0.1 → level = ceil(11*0.9)/10 = 10/10 = 1.0 → max
    vals = np.arange(10, dtype=float)
    q = _conformal_quantile(vals, alpha=0.1)
    assert q == pytest.approx(9.0)


def test_conformal_quantile_alpha_half() -> None:
    vals = np.arange(100, dtype=float)
    q = _conformal_quantile(vals, alpha=0.5)
    # level = ceil(101*0.5)/100 = 51/100 → quantile method=higher → ~50
    assert 48 <= q <= 52


def test_sweep_validity_synthetic() -> None:
    """Generate iid uniform K, m_omega; abstention rate should match guarantee."""
    rng = np.random.default_rng(0)
    n = 1000
    K = rng.uniform(0, 1, n)
    mo = rng.uniform(0, 1, n)
    y = rng.integers(0, 2, n)
    out = sweep_one(K, mo, y, alphas=[0.1, 0.2], k_folds=5, seed=0)
    for row in out:
        a = row["alpha"]
        emp = row["empirical_abstain_mean"]
        # On iid data, with two independent gates each at level α,
        # union bound: abst ≤ 2α; for independent gates: ≈ 1−(1−α)² ≈ 2α−α².
        # Allow a small slack for finite sample noise.
        assert a - 0.05 <= emp <= 2 * a + 0.05, (
            f"α={a}: empirical abstain {emp:.3f} outside [α-0.05, 2α+0.05]"
        )


def test_sweep_handles_small_run() -> None:
    """Small but valid run should still produce output."""
    rng = np.random.default_rng(1)
    K = rng.uniform(0, 1, 50)
    mo = rng.uniform(0, 1, 50)
    y = rng.integers(0, 2, 50)
    out = sweep_one(K, mo, y, alphas=[0.1], k_folds=5, seed=1)
    assert len(out) == 1
    assert out[0]["n_folds"] == 5
