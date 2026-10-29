"""Tests for the RIVF 2026 evidence profilers, evidential fusion and budget calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evidence import profilers as P
from src.evidence.fusion import apply_budget, budget_threshold, fuse, masses, weight


# ------------------------------------------------------------------ fusion algebra
def test_weights_add_under_dempster_for_same_focal_set():
    s1, s2 = 0.6, 0.3
    w = weight(s1) + weight(s2)
    # Dempster on two simple support functions for {err}: 1 - (1-s1)(1-s2)
    assert np.isclose(1 - np.exp(-w), 1 - (1 - s1) * (1 - s2))


def test_masses_sum_to_one_and_match_dempster_rule():
    rng = np.random.default_rng(0)
    S, T = rng.random(100), rng.random(100)
    wp, wn = weight(S), weight(T)
    m = masses(wp, wn)
    assert np.allclose(m.m_err + m.m_clean + m.m_omega, 1.0)
    # direct Dempster combination of m1({err})=S and m2({clean})=T
    K = S * T
    assert np.allclose(m.K, K)
    assert np.allclose(m.m_err, S * (1 - T) / (1 - K))
    assert np.allclose(m.m_clean, T * (1 - S) / (1 - K))


def test_score_is_normalised_plausibility():
    rng = np.random.default_rng(1)
    wp, wn = rng.random(50) * 3, rng.random(50) * 3
    m = masses(wp, wn)
    pl_err, pl_clean = m.m_err + m.m_omega, m.m_clean + m.m_omega
    assert np.allclose(m.score, pl_err / (pl_err + pl_clean))


def test_family_rule_is_cautious_within_and_dempster_across():
    n, m = 1, 1
    ev = {"pattern": np.full((n, m), 0.6), "unit": np.full((n, m), 0.4),   # syntax family
          "missing": np.zeros((n, m)), "typo": np.full((n, m), 0.5), "fd": np.zeros((n, m))}
    out = fuse(ev, P.FAMILY, None, rule="family", use_llm=False)
    expected = 1 - (1 - 0.6) * (1 - 0.5)  # max(0.6, 0.4) in syntax, 0.5 in value
    assert np.isclose(out["cell"].m_err[0, 0], expected)
    dem = fuse(ev, P.FAMILY, None, rule="dempster", use_llm=False)
    assert np.isclose(dem["cell"].m_err[0, 0], 1 - 0.4 * 0.6 * 0.5)
    cau = fuse(ev, P.FAMILY, None, rule="cautious", use_llm=False)
    assert np.isclose(cau["cell"].m_err[0, 0], 0.6)


# ------------------------------------------------------------------ budget guarantee
@pytest.mark.parametrize("beta", [0.1, 0.3, 0.5])
def test_review_budget_bound_holds_in_simulation(beta):
    rng = np.random.default_rng(42)
    n, reps = 99, 20000
    hits = 0
    for _ in range(reps):
        scores = np.round(rng.random(n + 1), 1)  # heavy ties on purpose
        thr = budget_threshold(scores[:n], beta, rng)
        hits += bool(apply_budget(scores[n:], thr, rng)[0])
    p = hits / reps
    se = np.sqrt(beta * (1 - beta) / reps)
    assert beta - 1 / (n + 1) - 4 * se < p <= beta + 4 * se


# ------------------------------------------------------------------ profilers
def test_pattern_rarity_flags_rare_shape_in_regular_column():
    df = pd.DataFrame({"zip": ["35233"] * 60 + ["3x233"]})
    E = P.pattern_rarity(df)
    assert E[-1, 0] > 0.5 and E[:-1, 0].max() == 0


def test_missing_tokens_ignore_mostly_null_columns():
    df = pd.DataFrame({"a": ["1"] * 9 + ["N/A"], "b": [""] * 8 + ["x", "y"]})
    E = P.missing_tokens(df)
    assert E[9, 0] == pytest.approx(0.9)
    assert E[:, 1].max() == 0  # 80% null: null is the convention


def test_typo_profiler_ignores_pure_digit_changes():
    df = pd.DataFrame({"city": ["birmingham"] * 10 + ["birminghxm"], "t": ["6:55 a.m."] * 10 + ["6:59 a.m."]})
    E = P.near_duplicate_typos(df)
    assert E[-1, 0] > 0.5
    assert E[-1, 1] == 0


def test_fd_violation_flags_minority_value():
    df = pd.DataFrame({"zip": ["1"] * 10 + ["2"] * 10, "city": ["a"] * 9 + ["b"] + ["c"] * 10})
    E = P.fd_violations(df)
    j = list(df.columns).index("city")
    assert E[9, j] > 0.8 and E[np.arange(20) != 9, j].max() == 0


def test_unit_inconsistency_flags_decorated_numbers():
    df = pd.DataFrame({"abv": ["0.05"] * 7 + ["0.06%"] * 3, "oz": ["12.0 oz", "12 ounce", "16 OZ.", "12.0 oz"] * 2 + ["12 oz", "16 oz."]})
    E = P.unit_inconsistency(df)
    assert (E[7:, 0] > 0).all() and E[:7, 0].max() == 0
    assert (E[:, 1] > 0).all()
