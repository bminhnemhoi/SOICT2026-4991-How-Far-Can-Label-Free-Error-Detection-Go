"""Conflict-Aware Decision Policy (CADP).

Decision rules (idea_main.md §4.6):
    error   if  Bel({err})   > tau_e  AND  K < tau_k  AND  m(Omega) < tau_Omega
    clean   if  Bel({clean}) > tau_c  AND  K < tau_k  AND  m(Omega) < tau_Omega
    abstain if  K >= tau_k   OR  m(Omega) >= tau_Omega  OR  neither threshold met
    escalate if  abstain  AND  re_prompt() still abstains  (handled at pipeline level)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from src.ds.combine import DSResult

Decision = Literal["error", "clean", "abstain"]


@dataclass(frozen=True)
class CADPThresholds:
    tau_e: float = 0.5  # min Bel({err}) to commit 'error'
    tau_c: float = 0.5  # min Bel({clean}) to commit 'clean'
    tau_k: float = 0.4  # max conflict K allowed
    tau_omega: float = 0.4  # max ignorance m(Omega) allowed

    def as_dict(self) -> dict[str, float]:
        return {"tau_e": self.tau_e, "tau_c": self.tau_c, "tau_k": self.tau_k, "tau_omega": self.tau_omega}


@dataclass(frozen=True)
class CADPDecision:
    decision: Decision
    p_hat: float  # Pl({err}) if error, Pl({clean}) if clean, else NaN
    reason: str  # short string for logging

    def as_dict(self) -> dict[str, object]:
        return {"decision": self.decision, "p_hat": self.p_hat, "reason": self.reason}


def decide(ds: DSResult, thr: CADPThresholds = CADPThresholds()) -> CADPDecision:
    """Apply CADP rules to a DSResult."""
    k_bad = ds.K >= thr.tau_k
    o_bad = ds.mass.m_omega >= thr.tau_omega
    if k_bad or o_bad:
        return CADPDecision(
            decision="abstain",
            p_hat=float("nan"),
            reason=f"K={ds.K:.3f}>=tau_k={thr.tau_k}" if k_bad else f"m_Omega={ds.mass.m_omega:.3f}>=tau_omega={thr.tau_omega}",
        )
    if ds.bel_err > thr.tau_e:
        return CADPDecision(decision="error", p_hat=ds.pl_err, reason=f"Bel_err={ds.bel_err:.3f}>tau_e={thr.tau_e}")
    if ds.bel_clean > thr.tau_c:
        return CADPDecision(decision="clean", p_hat=ds.pl_clean, reason=f"Bel_clean={ds.bel_clean:.3f}>tau_c={thr.tau_c}")
    return CADPDecision(
        decision="abstain",
        p_hat=float("nan"),
        reason=f"neither Bel>tau: err={ds.bel_err:.3f},cl={ds.bel_clean:.3f}",
    )
