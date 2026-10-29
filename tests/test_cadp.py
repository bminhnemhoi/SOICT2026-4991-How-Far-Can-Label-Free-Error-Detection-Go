"""Tests for src/cadp/policy.py and src/cadp/conformal.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.bcl.bcl import MassFunction
from src.cadp.conformal import calibrate_thresholds
from src.cadp.policy import CADPThresholds, decide
from src.ds.combine import ds_combine


def _ds(ae: float, ac: float, ao: float, ee: float, ec: float, eo: float):
    return ds_combine(MassFunction(ae, ac, ao), MassFunction(ee, ec, eo))


def test_decide_error_clear():
    r = _ds(0.9, 0.05, 0.05, 0.9, 0.05, 0.05)
    d = decide(r, CADPThresholds(tau_e=0.5, tau_c=0.5, tau_k=0.4, tau_omega=0.4))
    assert d.decision == "error"
    assert d.p_hat > 0.9


def test_decide_clean_clear():
    r = _ds(0.05, 0.9, 0.05, 0.05, 0.9, 0.05)
    d = decide(r)
    assert d.decision == "clean"


def test_decide_abstain_high_conflict():
    """Two strongly opposing sources → K close to cap."""
    r = _ds(0.5, 0.3, 0.2, 0.3, 0.5, 0.2)  # K = 0.5*0.5 + 0.3*0.3 = 0.34
    d = decide(r, CADPThresholds(tau_k=0.3))
    assert d.decision == "abstain"
    assert "K=" in d.reason


def test_decide_abstain_high_omega():
    r = _ds(0.1, 0.1, 0.8, 0.1, 0.1, 0.8)
    d = decide(r, CADPThresholds(tau_omega=0.4))
    assert d.decision == "abstain"
    assert "Omega" in d.reason or "omega" in d.reason


def test_decide_abstain_no_bel_above_threshold():
    r = _ds(0.3, 0.3, 0.4, 0.3, 0.3, 0.4)
    d = decide(r, CADPThresholds(tau_e=0.9, tau_c=0.9, tau_k=0.5, tau_omega=0.9))
    assert d.decision == "abstain"


def test_conformal_calibration_basic():
    """Synthetic cal set: separable err/clean with some noisy middle."""
    rng = np.random.default_rng(0)
    n = 200
    # Half err, half clean
    y = np.array(["err"] * (n // 2) + ["clean"] * (n // 2))
    # err samples: bel_err high, K low
    bel_err = np.concatenate([rng.beta(8, 2, n // 2), rng.beta(2, 8, n // 2)])
    bel_clean = np.concatenate([rng.beta(2, 8, n // 2), rng.beta(8, 2, n // 2)])
    K = rng.uniform(0, 0.3, n)
    m_omega = rng.uniform(0, 0.3, n)
    cal = calibrate_thresholds(K, m_omega, bel_err, bel_clean, y, alpha=0.1, lambda_=1.0, precision_floor=0.6)
    assert 0 < cal.thresholds.tau_k < 1
    assert 0 < cal.thresholds.tau_omega < 1
    assert 0 < cal.thresholds.tau_e < 1
    assert cal.precision_err >= 0.6
    assert cal.f1_err > 0.4
