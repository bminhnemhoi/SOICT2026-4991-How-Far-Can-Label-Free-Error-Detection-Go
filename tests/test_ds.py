"""Tests for src/ds/combine.py."""

from __future__ import annotations

import pytest

from src.bcl.bcl import MassFunction
from src.ds.combine import ds_combine


def test_agreement_on_err_reinforces():
    m_a = MassFunction(m_err=0.8, m_clean=0.1, m_omega=0.1)
    m_e = MassFunction(m_err=0.8, m_clean=0.1, m_omega=0.1)
    r = ds_combine(m_a, m_e)
    # K = 0.8*0.1 + 0.1*0.8 = 0.16
    assert r.K == pytest.approx(0.16, abs=1e-9)
    # m_err_num   = 0.8*0.8 + 0.8*0.1 + 0.1*0.8 = 0.80
    # m_clean_num = 0.1*0.1 + 0.1*0.1 + 0.1*0.1 = 0.03
    # m_omega_num = 0.1 * 0.1 = 0.01
    assert r.mass.m_err == pytest.approx(0.80 / 0.84, abs=1e-6)
    assert r.mass.m_clean == pytest.approx(0.03 / 0.84, abs=1e-6)
    assert r.mass.m_omega == pytest.approx(0.01 / 0.84, abs=1e-6)
    assert r.bel_err == r.mass.m_err
    assert r.pl_err == pytest.approx(r.mass.m_err + r.mass.m_omega, abs=1e-9)


def test_total_conflict_raises():
    m_a = MassFunction(m_err=1.0, m_clean=0.0, m_omega=0.0)
    m_e = MassFunction(m_err=0.0, m_clean=1.0, m_omega=0.0)
    # K = 1.0 → combination undefined
    with pytest.raises(ValueError):
        ds_combine(m_a, m_e)


def test_vacuous_prior_is_identity():
    """Combining with m_E = (0, 0, 1) (total ignorance) returns m_A."""
    m_a = MassFunction(m_err=0.3, m_clean=0.5, m_omega=0.2)
    m_e = MassFunction(m_err=0.0, m_clean=0.0, m_omega=1.0)
    r = ds_combine(m_a, m_e)
    assert r.K == pytest.approx(0.0, abs=1e-9)
    assert r.mass.m_err == pytest.approx(0.3, abs=1e-9)
    assert r.mass.m_clean == pytest.approx(0.5, abs=1e-9)
    assert r.mass.m_omega == pytest.approx(0.2, abs=1e-9)


def test_partial_conflict_textbook():
    """Textbook example: two half-confident opposing sources."""
    m_a = MassFunction(m_err=0.6, m_clean=0.2, m_omega=0.2)
    m_e = MassFunction(m_err=0.2, m_clean=0.6, m_omega=0.2)
    r = ds_combine(m_a, m_e)
    # K = 0.6*0.6 + 0.2*0.2 = 0.36 + 0.04 = 0.40
    assert r.K == pytest.approx(0.40, abs=1e-9)
    # m_err_num = 0.12 + 0.12 + 0.04 = 0.28 → /0.60 ≈ 0.4667
    assert r.mass.m_err == pytest.approx(0.28 / 0.60, abs=1e-6)
    assert r.mass.m_clean == pytest.approx(0.28 / 0.60, abs=1e-6)
    assert r.mass.m_omega == pytest.approx(0.04 / 0.60, abs=1e-6)


def test_bel_and_pl_consistent():
    m_a = MassFunction(m_err=0.5, m_clean=0.3, m_omega=0.2)
    m_e = MassFunction(m_err=0.4, m_clean=0.3, m_omega=0.3)
    r = ds_combine(m_a, m_e)
    assert r.bel_err <= r.pl_err
    assert r.bel_clean <= r.pl_clean
    assert r.pl_err + r.pl_clean >= 1.0  # because both include m_omega
