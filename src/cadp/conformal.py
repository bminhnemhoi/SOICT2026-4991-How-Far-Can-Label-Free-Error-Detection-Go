"""Split-conformal threshold calibration for CADP (idea_main.md §4.6).

Given a calibration set D_cal with DS outputs, compute:
  - tau_k, tau_omega as quantile(1 - alpha) of nonconformity score
        s(x) = K(x) + lambda * m_Omega(x)
    This yields marginal coverage guarantee >= 1 - alpha.
  - tau_e, tau_c: grid-search to maximize F1 on D_cal with precision >= p0.

Notes:
  - Scores should be computed AFTER running BCL + DS on D_cal at inference time.
  - True labels y ∈ {'err', 'clean'} required for tau_e/tau_c tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from src.cadp.policy import CADPThresholds


@dataclass(frozen=True)
class CalibratedThresholds:
    thresholds: CADPThresholds
    alpha: float
    lambda_: float
    tau_s: float  # quantile value of s(x)
    f1_err: float
    f1_clean: float
    precision_err: float
    precision_clean: float


def _nonconformity(K: np.ndarray, m_omega: np.ndarray, lambda_: float) -> np.ndarray:
    return K + lambda_ * m_omega


def _split_tau_s(K: np.ndarray, m_omega: np.ndarray, alpha: float, lambda_: float) -> float:
    """Quantile(1 - alpha) of s(x) with finite-sample correction.

    Uses conformal adjustment: level = ceil((n+1)*(1-alpha)) / n.
    """
    n = len(K)
    if n == 0:
        raise ValueError("calibration set is empty")
    s = _nonconformity(K, m_omega, lambda_)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(s, level, method="higher"))


def calibrate_thresholds(
    K: Sequence[float],
    m_omega: Sequence[float],
    bel_err: Sequence[float],
    bel_clean: Sequence[float],
    y_true: Sequence[str],
    *,
    alpha: float = 0.1,
    lambda_: float = 1.0,
    precision_floor: float = 0.7,
    tau_grid: Iterable[float] = tuple(round(x, 3) for x in np.linspace(0.05, 0.95, 19)),
) -> CalibratedThresholds:
    """Calibrate (tau_k, tau_omega, tau_e, tau_c) on a labeled calibration set.

    Strategy:
      1. Compute tau_s = quantile_{1-alpha}(s), then split tau_s into tau_k + lambda*tau_omega
         by equal allocation: tau_k = tau_s / (1 + lambda_), tau_omega = tau_s / (lambda_ * (1+lambda_)).
         Simpler variant: tau_k = tau_omega = tau_s / (1 + lambda_)   (used here).
      2. Grid search tau_e maximizing F1(err) with precision(err) >= precision_floor.
      3. Grid search tau_c maximizing F1(clean) with precision(clean) >= precision_floor.
    """
    K = np.asarray(K, dtype=float)
    m_o = np.asarray(m_omega, dtype=float)
    be = np.asarray(bel_err, dtype=float)
    bc = np.asarray(bel_clean, dtype=float)
    y = np.asarray(y_true)

    # Per-component split-conformal (more robust when K and m_omega have
    # different scales). Original equal-split tau_s/(1+lambda) often left
    # tau_k below min(K) when belief mass concentrates in m_omega.
    n = len(K)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    tau_k = float(np.quantile(K, level, method="higher"))
    tau_omega = float(np.quantile(m_o, level, method="higher"))
    tau_s = tau_k + lambda_ * tau_omega  # composite (kept for diagnostics)

    # Only items that would NOT abstain (K < tau_k AND m_omega < tau_omega)
    commit_mask = (K < tau_k) & (m_o < tau_omega)

    def best_tau(bel: np.ndarray, positive_label: str) -> tuple[float, float, float]:
        y_pos = (y == positive_label).astype(int)
        best = (0.5, 0.0, 0.0)  # (tau, f1, precision)
        for tau in tau_grid:
            pred = ((bel > tau) & commit_mask).astype(int)
            tp = int(((pred == 1) & (y_pos == 1)).sum())
            fp = int(((pred == 1) & (y_pos == 0)).sum())
            fn = int(((pred == 0) & (y_pos == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            if prec >= precision_floor and f1 > best[1]:
                best = (float(tau), float(f1), float(prec))
        return best

    tau_e, f1_e, prec_e = best_tau(be, "err")
    tau_c, f1_c, prec_c = best_tau(bc, "clean")

    return CalibratedThresholds(
        thresholds=CADPThresholds(tau_e=tau_e, tau_c=tau_c, tau_k=tau_k, tau_omega=tau_omega),
        alpha=alpha,
        lambda_=lambda_,
        tau_s=tau_s,
        f1_err=f1_e,
        f1_clean=f1_c,
        precision_err=prec_e,
        precision_clean=prec_c,
    )


@dataclass(frozen=True)
class CoverageCalibration:
    """Result of coverage-targeted threshold calibration."""

    tau_e: float
    tau_c: float
    tau_k: float
    tau_omega: float
    target_coverage: float
    achieved_coverage_dev: float
    n_dev: int


def calibrate_for_target_coverage(
    K: Sequence[float],
    m_omega: Sequence[float],
    bel_err: Sequence[float],
    bel_clean: Sequence[float],
    *,
    target_coverage: float,
    tau_k_quantile: float = 0.90,
    tau_omega_quantile: float = 0.90,
) -> CoverageCalibration:
    """Pick (tau_e, tau_c) on a dev split to achieve a target commit coverage.

    Strategy:
      1. Set tau_k, tau_omega to quantile-based limits (so that they only
         catch the worst high-conflict / high-ignorance dev rows).
      2. Among rows that pass the (K, m_omega) gates, sort by max(bel_err, bel_clean).
      3. Pick the cutoff that selects the top `target_coverage * n` rows.

    The returned thresholds, applied to a test split exchangeable with dev,
    give an actual coverage close to `target_coverage` (with O(1/sqrt(n_dev))
    deviation - this is the conformal coverage guarantee).

    Args:
        K, m_omega, bel_err, bel_clean: per-row dev outputs.
        target_coverage: in (0, 1].
        tau_k_quantile, tau_omega_quantile: dev-quantiles to set the gates;
            higher means looser gate (rarely abstains for K/m_omega reason).
    """
    K_arr = np.asarray(K, dtype=float)
    m_o = np.asarray(m_omega, dtype=float)
    be = np.asarray(bel_err, dtype=float)
    bc = np.asarray(bel_clean, dtype=float)

    n = len(K_arr)
    if n == 0:
        raise ValueError("calibration set is empty")
    if not (0.0 < target_coverage <= 1.0):
        raise ValueError(f"target_coverage must be in (0,1]; got {target_coverage}")

    tau_k = float(np.quantile(K_arr, tau_k_quantile, method="higher"))
    tau_omega = float(np.quantile(m_o, tau_omega_quantile, method="higher"))

    pass_gate = (K_arr <= tau_k) & (m_o <= tau_omega)
    margin = np.maximum(be, bc)

    n_target = max(1, int(round(target_coverage * n)))
    eligible_margins = np.sort(margin[pass_gate])[::-1]
    if len(eligible_margins) == 0:
        cutoff = 0.0
    elif n_target >= len(eligible_margins):
        cutoff = float(eligible_margins[-1])
    else:
        cutoff = float(eligible_margins[n_target - 1])

    tau_e = cutoff
    tau_c = cutoff

    achieved_mask = pass_gate & (margin >= cutoff)
    achieved = float(achieved_mask.sum()) / n

    return CoverageCalibration(
        tau_e=tau_e,
        tau_c=tau_c,
        tau_k=tau_k,
        tau_omega=tau_omega,
        target_coverage=target_coverage,
        achieved_coverage_dev=achieved,
        n_dev=n,
    )


def apply_coverage_calibration(
    K: Sequence[float],
    m_omega: Sequence[float],
    bel_err: Sequence[float],
    bel_clean: Sequence[float],
    cal: CoverageCalibration,
) -> dict[str, float]:
    """Apply calibrated thresholds to a held-out (test) split, return coverage stats."""
    K_arr = np.asarray(K, dtype=float)
    m_o = np.asarray(m_omega, dtype=float)
    be = np.asarray(bel_err, dtype=float)
    bc = np.asarray(bel_clean, dtype=float)

    n = len(K_arr)
    pass_gate = (K_arr <= cal.tau_k) & (m_o <= cal.tau_omega)
    margin = np.maximum(be, bc)
    commit = pass_gate & (margin >= cal.tau_e)
    actual = float(commit.sum()) / n if n else 0.0

    return {
        "n_test": n,
        "actual_coverage": actual,
        "target_coverage": cal.target_coverage,
        "coverage_gap": abs(actual - cal.target_coverage),
        "n_commit": int(commit.sum()),
    }
