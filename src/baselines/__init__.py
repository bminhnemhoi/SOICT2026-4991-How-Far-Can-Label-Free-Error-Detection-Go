"""Lightweight baselines for MA-LED comparison.

Each baseline takes the SAME (auditor_output, expert_output) pair produced by
the runner and returns a decision record. This lets us re-use the agent outputs
from an MA-LED run without re-calling the LLM, saving cost on ablations.

Baselines:
  - single_llm:     use only Auditor's label (Expert ignored). No DS.
  - majority_vote:  A and E vote; tie -> 'abstain'.
  - debate_no_ds:   weighted average of (p,c); no DS fusion.
  - bcl_ds_only:    MA-LED core (BCL+DS) WITHOUT CADP conflict gate.
"""

from __future__ import annotations

from typing import Any

from src.bcl.bcl import BCLWeights, build_mass
from src.ds.combine import ds_combine


def _p_from_agent(out: Any) -> float:
    p = float(out.p)
    return p if out.label == "err" else 1.0 - p


def single_llm(a_out: Any, e_out: Any) -> dict[str, Any]:
    p_err = _p_from_agent(a_out)
    if a_out.label == "err" and a_out.p >= 0.5:
        dec = "error"
    elif a_out.label == "clean" and a_out.p >= 0.5:
        dec = "clean"
    else:
        dec = "abstain"
    return {"decision": dec, "p_hat": p_err, "reason": f"single_llm(auditor):{a_out.label}@{a_out.p:.2f}"}


def majority_vote(a_out: Any, e_out: Any) -> dict[str, Any]:
    p_err = 0.5 * (_p_from_agent(a_out) + _p_from_agent(e_out))
    if a_out.label == e_out.label:
        dec = "error" if a_out.label == "err" else "clean"
    else:
        dec = "abstain"
    return {"decision": dec, "p_hat": p_err, "reason": f"vote(A={a_out.label},E={e_out.label})"}


def debate_no_ds(a_out: Any, e_out: Any, w_a: float = 0.5, w_b: float = 0.5) -> dict[str, Any]:
    p_err = w_a * _p_from_agent(a_out) + w_b * _p_from_agent(e_out)
    if p_err >= 0.6:
        dec = "error"
    elif p_err <= 0.4:
        dec = "clean"
    else:
        dec = "abstain"
    return {"decision": dec, "p_hat": p_err, "reason": f"debate_no_ds:p_err={p_err:.2f}"}


def bcl_ds_only(
    a_out: Any, e_out: Any, weights: BCLWeights,
    tau_e: float = 0.5, tau_c: float = 0.5,
) -> dict[str, Any]:
    m_A = build_mass(a_out.label, a_out.p, a_out.c, a_out.a, weights=weights)
    m_E = build_mass(e_out.label, e_out.p, e_out.c, e_out.a, weights=weights)
    ds = ds_combine(m_A, m_E)
    if ds.bel_err >= tau_e:
        dec = "error"
    elif ds.bel_clean >= tau_c:
        dec = "clean"
    else:
        dec = "abstain"
    return {"decision": dec, "p_hat": ds.bel_err, "reason": f"bcl_ds_only:K={ds.K:.2f}"}


BASELINES = {
    "single_llm": single_llm,
    "majority_vote": majority_vote,
    "debate_no_ds": debate_no_ds,
    "bcl_ds_only": bcl_ds_only,
}
