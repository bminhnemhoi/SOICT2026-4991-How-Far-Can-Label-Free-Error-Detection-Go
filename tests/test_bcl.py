"""Tests for src/bcl/bcl.py."""

from __future__ import annotations

import math

import pytest

from src.bcl.bcl import BCLWeights, MassFunction, build_mass


def test_mass_sums_to_one_generic():
    m = build_mass(yhat="err", p=0.8, c=0.7, a=0.9)
    assert abs(m.m_err + m.m_clean + m.m_omega - 1.0) < 1e-9
    assert 0 <= m.m_err <= 1 and 0 <= m.m_clean <= 1 and 0 <= m.m_omega <= 1


def test_high_confidence_err_pushes_mass_to_err():
    m = build_mass(yhat="err", p=1.0, c=1.0, a=1.0)
    # No ignorance signal (all perfect); s_unk = 0; s_err > s_clean.
    assert m.m_err > m.m_clean
    assert m.m_err > m.m_omega


def test_high_confidence_clean_pushes_mass_to_clean():
    m = build_mass(yhat="clean", p=1.0, c=1.0, a=1.0)
    assert m.m_clean > m.m_err
    assert m.m_clean > m.m_omega


def test_low_signals_produce_high_omega():
    """When p,c,a are all low, ignorance trigger g = max(1-p,1-c,1-a) is high → m(Omega) dominates."""
    m = build_mass(yhat="err", p=0.05, c=0.05, a=0.05)
    assert m.m_omega > m.m_err
    assert m.m_omega > m.m_clean


def test_full_ignorance_edge_p_1_c_1_a_1_no_label():
    """All signals perfect but yhat='err' → indicator makes clean logit = w1*phi(0) only."""
    m = build_mass(yhat="err", p=1.0, c=1.0, a=1.0)
    # With identity maps and default weights, s_err = 1 + 1 + 1 = 3; s_clean = 0 + 1 + 1 = 2; s_unk = 0.
    # Softmax(3,2,0).
    denom = math.exp(3) + math.exp(2) + math.exp(0)
    assert abs(m.m_err - math.exp(3) / denom) < 1e-9
    assert abs(m.m_clean - math.exp(2) / denom) < 1e-9


def test_invalid_label_raises():
    with pytest.raises(ValueError):
        build_mass(yhat="unknown", p=0.5, c=0.5, a=0.5)


def test_out_of_range_signal_raises():
    with pytest.raises(ValueError):
        build_mass(yhat="err", p=1.2, c=0.5, a=0.5)


def test_weights_zero_on_w4_removes_ignorance():
    w = BCLWeights(w1=1.0, w2=1.0, w3=1.0, w4=0.0)
    m = build_mass(yhat="err", p=0.1, c=0.1, a=0.1, weights=w)
    # s_unk = 0 now; but logits (all small). Still softmax 3-way.
    assert m.m_err + m.m_clean + m.m_omega == pytest.approx(1.0, abs=1e-9)


def test_mass_function_rejects_invalid_sum():
    with pytest.raises(ValueError):
        MassFunction(m_err=0.5, m_clean=0.3, m_omega=0.3)  # sums to 1.1
