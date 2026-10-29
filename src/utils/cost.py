"""Cost estimation + tracking for LLM runs.

Pricing is read from env (MA_LED_PRICE_IN_PER_1M / MA_LED_PRICE_OUT_PER_1M).
Default = gpt-4o-mini pricing (2025): $0.15 in, $0.60 out per 1M tokens.

Rule (docs/12-cost-and-api-rules.md):
- ESTIMATE BEFORE every run. scripts/run.sh aborts if no cost_estimate.json.
- Log actual/estimate ratio in report.md — must be ∈ [0.7, 1.5].
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Any


def _price_in() -> float:
    return float(os.environ.get("MA_LED_PRICE_IN_PER_1M", "0.15"))


def _price_out() -> float:
    return float(os.environ.get("MA_LED_PRICE_OUT_PER_1M", "0.60"))


def usd_for(tokens_in: int, tokens_out: int) -> float:
    return tokens_in / 1_000_000 * _price_in() + tokens_out / 1_000_000 * _price_out()


@dataclass
class CostTracker:
    """Accumulate tokens/USD/calls for 1 run. Dump to cost.json at end."""

    tokens_in: int = 0
    tokens_out: int = 0
    n_calls: int = 0
    n_cache_hits: int = 0
    by_provider: dict[str, dict[str, int]] = field(default_factory=dict)

    def record(
        self,
        tokens_in: int,
        tokens_out: int,
        provider: str = "openai",
        cache_hit: bool = False,
    ) -> None:
        if cache_hit:
            self.n_cache_hits += 1
            return
        self.tokens_in += tokens_in
        self.tokens_out += tokens_out
        self.n_calls += 1
        p = self.by_provider.setdefault(
            provider, {"tokens_in": 0, "tokens_out": 0, "n_calls": 0}
        )
        p["tokens_in"] += tokens_in
        p["tokens_out"] += tokens_out
        p["n_calls"] += 1

    @property
    def usd(self) -> float:
        return round(usd_for(self.tokens_in, self.tokens_out), 6)

    def as_dict(self, run_id: str | None = None) -> dict[str, Any]:
        d = asdict(self)
        # Add USD per provider
        by_p = {}
        for name, v in d["by_provider"].items():
            by_p[name] = {**v, "usd": round(usd_for(v["tokens_in"], v["tokens_out"]), 6)}
        d["by_provider"] = by_p
        d["usd"] = self.usd
        if run_id:
            d = {"run_id": run_id, **d}
        return d


def estimate_cost(
    n_samples: int,
    tokens_in_per_call: int,
    tokens_out_per_call: int,
    n_agents: int = 3,
    self_consistency_m: int = 1,
    n_perturb: int = 0,
    fudge_factor: float = 1.3,
) -> dict[str, Any]:
    """Pre-run cost estimate.

    total_calls = n_samples * n_agents * self_consistency_m * (1 + n_perturb)
    """
    total_calls = n_samples * n_agents * self_consistency_m * (1 + n_perturb)
    est_in = int(total_calls * tokens_in_per_call * fudge_factor)
    est_out = int(total_calls * tokens_out_per_call * fudge_factor)
    est_usd = round(usd_for(est_in, est_out), 4)
    return {
        "n_samples": n_samples,
        "n_agents": n_agents,
        "self_consistency_m": self_consistency_m,
        "n_perturb": n_perturb,
        "tokens_in_per_call": tokens_in_per_call,
        "tokens_out_per_call": tokens_out_per_call,
        "fudge_factor": fudge_factor,
        "total_calls": total_calls,
        "estimated_tokens_in": est_in,
        "estimated_tokens_out": est_out,
        "estimated_usd": est_usd,
        "pricing": {"input_per_1m": _price_in(), "output_per_1m": _price_out()},
    }
