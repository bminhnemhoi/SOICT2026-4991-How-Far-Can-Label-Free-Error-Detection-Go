"""Belief Construction Layer (BCL).

Maps per-agent signals (yhat, p, c, a) -> mass function over frame
Omega = {err, clean} with ignorance mass m(Omega).

Formulas (idea_main.md §4.3):
    s_err   = w1 * phi(p * 1[yhat=err])   + w2 * psi(c) + w3 * rho(a)
    s_clean = w1 * phi(p * 1[yhat=clean]) + w2 * psi(c) + w3 * rho(a)
    s_unk   = w4 * g(1-p, 1-c, 1-a)             # ignorance trigger
    m = softmax([s_err, s_clean, s_unk])
    m({err}), m({clean}), m(Omega) = m[0], m[1], m[2]

phi, psi, rho default to identity (Platt scaling fit later on calib split).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

Label = Literal["err", "clean"]
MonotoneMap = Callable[[float], float]


def _identity(x: float) -> float:
    return float(x)


def _softmax3(a: float, b: float, c: float) -> tuple[float, float, float]:
    m = max(a, b, c)
    ea, eb, ec = np.exp(a - m), np.exp(b - m), np.exp(c - m)
    s = ea + eb + ec
    return float(ea / s), float(eb / s), float(ec / s)


@dataclass(frozen=True)
class BCLWeights:
    """Weights for BCL logits (see idea_main.md §4.3)."""

    w1: float = 1.0  # confidence
    w2: float = 1.0  # explanation-evidence consistency
    w3: float = 1.0  # self-agreement
    w4: float = 1.0  # ignorance trigger

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.w1, self.w2, self.w3, self.w4)


@dataclass(frozen=True)
class MassFunction:
    """Mass function on Omega = {err, clean}.

    Invariants: all >= 0, sum == 1.
    """

    m_err: float
    m_clean: float
    m_omega: float

    def __post_init__(self) -> None:
        total = self.m_err + self.m_clean + self.m_omega
        if not (abs(total - 1.0) < 1e-6):
            raise ValueError(f"mass must sum to 1, got {total}")
        for v, name in ((self.m_err, "m_err"), (self.m_clean, "m_clean"), (self.m_omega, "m_omega")):
            if v < -1e-9 or v > 1 + 1e-9:
                raise ValueError(f"{name}={v} not in [0,1]")

    def as_dict(self) -> dict[str, float]:
        return {"m_err": self.m_err, "m_clean": self.m_clean, "m_omega": self.m_omega}


def build_mass(
    yhat: Label,
    p: float,
    c: float,
    a: float,
    weights: BCLWeights = BCLWeights(),
    phi: MonotoneMap = _identity,
    psi: MonotoneMap = _identity,
    rho: MonotoneMap = _identity,
) -> MassFunction:
    """Build a single-agent mass function from raw signals.

    Args:
        yhat: agent's predicted label, 'err' or 'clean'.
        p: model confidence in [0, 1].
        c: explanation-evidence consistency in [0, 1].
        a: self-agreement in [0, 1].
        weights: BCLWeights.
        phi, psi, rho: calibrated monotone transforms (default identity).
    """
    if yhat not in ("err", "clean"):
        raise ValueError(f"yhat must be 'err' or 'clean', got {yhat!r}")
    for name, v in (("p", p), ("c", c), ("a", a)):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name}={v} out of [0,1]")

    w1, w2, w3, w4 = weights.as_tuple()
    p_err = p if yhat == "err" else 0.0
    p_clean = p if yhat == "clean" else 0.0

    s_err = w1 * phi(p_err) + w2 * psi(c) + w3 * rho(a)
    s_clean = w1 * phi(p_clean) + w2 * psi(c) + w3 * rho(a)
    s_unk = w4 * max(1.0 - p, 1.0 - c, 1.0 - a)

    m_err, m_clean, m_omega = _softmax3(s_err, s_clean, s_unk)
    return MassFunction(m_err=m_err, m_clean=m_clean, m_omega=m_omega)
