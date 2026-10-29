"""Dempster-Shafer combination on frame Omega = {err, clean}.

Focal elements: {err}, {clean}, Omega.

Combination (idea_main.md §4.4):
    K = m_A({err})*m_E({clean}) + m_A({clean})*m_E({err})        (conflict)
    m({err})   = [ m_A.err*m_E.err  + m_A.err*m_E.O   + m_A.O*m_E.err  ] / (1-K)
    m({clean}) = [ m_A.cl*m_E.cl    + m_A.cl*m_E.O    + m_A.O*m_E.cl   ] / (1-K)
    m(Omega)   = m_A.O * m_E.O / (1-K)

Belief / Plausibility:
    Bel({err})   = m({err});    Pl({err})   = m({err})   + m(Omega)
    Bel({clean}) = m({clean});  Pl({clean}) = m({clean}) + m(Omega)
"""

from __future__ import annotations

from dataclasses import dataclass

from src.bcl.bcl import MassFunction


@dataclass(frozen=True)
class DSResult:
    mass: MassFunction
    K: float  # conflict
    bel_err: float
    bel_clean: float
    pl_err: float
    pl_clean: float

    def as_dict(self) -> dict[str, float]:
        return {
            **self.mass.as_dict(),
            "K": self.K,
            "bel_err": self.bel_err,
            "bel_clean": self.bel_clean,
            "pl_err": self.pl_err,
            "pl_clean": self.pl_clean,
        }


def ds_combine(m_a: MassFunction, m_e: MassFunction, *, conflict_tol: float = 1 - 1e-9) -> DSResult:
    """Combine two mass functions via Dempster's rule.

    Raises:
        ValueError: if total conflict K >= conflict_tol (fusion undefined).
    """
    ae, ac, ao = m_a.m_err, m_a.m_clean, m_a.m_omega
    ee, ec, eo = m_e.m_err, m_e.m_clean, m_e.m_omega

    k = ae * ec + ac * ee
    if k >= conflict_tol:
        raise ValueError(f"Total conflict K={k:.6f} >= {conflict_tol}; DS combination undefined.")

    denom = 1.0 - k
    m_err = (ae * ee + ae * eo + ao * ee) / denom
    m_clean = (ac * ec + ac * eo + ao * ec) / denom
    m_omega = (ao * eo) / denom

    # Re-normalize tiny FP drift to preserve MassFunction invariant.
    s = m_err + m_clean + m_omega
    m_err, m_clean, m_omega = m_err / s, m_clean / s, m_omega / s

    mass = MassFunction(m_err=m_err, m_clean=m_clean, m_omega=m_omega)
    return DSResult(
        mass=mass,
        K=float(k),
        bel_err=m_err,
        bel_clean=m_clean,
        pl_err=m_err + m_omega,
        pl_clean=m_clean + m_omega,
    )
