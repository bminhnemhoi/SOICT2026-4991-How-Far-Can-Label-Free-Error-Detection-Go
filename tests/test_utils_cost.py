"""Tests for src/utils/cost.py"""

from __future__ import annotations

from src.utils.cost import CostTracker, estimate_cost, usd_for


def test_usd_for_pricing_default(monkeypatch):
    # Force default pricing.
    monkeypatch.setenv("MA_LED_PRICE_IN_PER_1M", "0.15")
    monkeypatch.setenv("MA_LED_PRICE_OUT_PER_1M", "0.60")
    # 1M in + 1M out → 0.15 + 0.60 = 0.75
    assert abs(usd_for(1_000_000, 1_000_000) - 0.75) < 1e-9


def test_cost_tracker_records_and_cache():
    t = CostTracker()
    t.record(1000, 500, provider="openai")
    t.record(0, 0, provider="openai", cache_hit=True)
    assert t.tokens_in == 1000
    assert t.tokens_out == 500
    assert t.n_calls == 1
    assert t.n_cache_hits == 1
    assert t.usd > 0


def test_estimate_cost_schema():
    est = estimate_cost(
        n_samples=100,
        tokens_in_per_call=1000,
        tokens_out_per_call=200,
        n_agents=3,
        self_consistency_m=1,
        n_perturb=0,
    )
    assert est["total_calls"] == 100 * 3
    assert est["estimated_usd"] > 0
    # Perturbation multiplies calls
    est2 = estimate_cost(100, 1000, 200, n_agents=3, self_consistency_m=1, n_perturb=5)
    assert est2["total_calls"] == 100 * 3 * 6
