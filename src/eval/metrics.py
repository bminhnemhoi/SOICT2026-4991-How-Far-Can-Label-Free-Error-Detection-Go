"""Evaluation metrics — detection / calibration / selective prediction.

Pure functions; no dependency on src/ internals. All inputs are arrays.
Schema for CSV output is in docs/06-metrics-spec.md.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)


def detection_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Binary detection metrics at given threshold + AUC-PR/ROC."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    # Single-class guards
    try:
        auc_pr = float(average_precision_score(y_true, y_score))
    except ValueError:
        auc_pr = float("nan")
    try:
        auc_roc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc_roc = float("nan")
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
    }


def expected_calibration_error(
    y_true: np.ndarray, y_conf: np.ndarray, n_bins: int = 10
) -> float:
    """ECE with equal-width bins on [0, 1]."""
    y_true = np.asarray(y_true).astype(int)
    y_conf = np.asarray(y_conf).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_conf >= lo) & (y_conf < hi if i < n_bins - 1 else y_conf <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = y_conf[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return float(ece)


def reliability_bins(
    y_true: np.ndarray, y_conf: np.ndarray, n_bins: int = 10
) -> list[dict[str, float]]:
    """Per-bin stats for reliability diagram."""
    y_true = np.asarray(y_true).astype(int)
    y_conf = np.asarray(y_conf).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_conf >= lo) & (y_conf < hi if i < n_bins - 1 else y_conf <= hi)
        out.append(
            {
                "bin_idx": i,
                "bin_lower": float(lo),
                "bin_upper": float(hi),
                "n_in_bin": int(mask.sum()),
                "mean_conf": float(y_conf[mask].mean()) if mask.any() else 0.0,
                "accuracy": float(y_true[mask].mean()) if mask.any() else 0.0,
            }
        )
    return out


def brier(y_true: np.ndarray, y_conf: np.ndarray) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_conf = np.asarray(y_conf).astype(float)
    return float(brier_score_loss(y_true, y_conf))


def risk_coverage_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_conf: np.ndarray,
    threshold: float = 0.5,
) -> list[dict[str, float]]:
    """Selective prediction: sort by confidence desc; at each coverage, compute risk (error rate).

    Risk = fraction of wrong predictions among samples retained.
    Returns 101 points (coverage = 0.00..1.00 in 0.01 steps).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_conf = np.asarray(y_conf).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    order = np.argsort(-y_conf)  # high confidence first
    y_true_s = y_true[order]
    y_pred_s = y_pred[order]
    n = len(y_true)
    wrong_cum = np.cumsum(y_true_s != y_pred_s)
    out = []
    for cov_pct in range(0, 101):
        k = max(1, int(np.ceil(n * cov_pct / 100))) if cov_pct > 0 else 0
        if k == 0:
            out.append({"coverage": 0.0, "risk": 0.0})
            continue
        risk = wrong_cum[k - 1] / k
        out.append({"coverage": k / n, "risk": float(risk)})
    return out


def aurc(rc_curve: list[dict[str, float]]) -> float:
    """Area Under Risk-Coverage curve (trapezoidal)."""
    xs = np.array([p["coverage"] for p in rc_curve])
    ys = np.array([p["risk"] for p in rc_curve])
    return float(np.trapezoid(ys, xs))


def risk_at_coverage(
    rc_curve: list[dict[str, float]], target_coverage: float
) -> float:
    """Linear interpolate risk at given coverage."""
    xs = np.array([p["coverage"] for p in rc_curve])
    ys = np.array([p["risk"] for p in rc_curve])
    return float(np.interp(target_coverage, xs, ys))


def abstention_quality(
    y_error: np.ndarray, abstain: np.ndarray
) -> dict[str, float]:
    """Treat abstain==1 as "flagged for human review". Error==1 ground truth.
    Good system: high precision + recall of abstain covering true errors.
    """
    y_error = np.asarray(y_error).astype(int)
    abstain = np.asarray(abstain).astype(int)
    p, r, f, _ = precision_recall_fscore_support(
        y_error, abstain, average="binary", zero_division=0
    )
    coverage = float(1 - abstain.mean())
    return {
        "coverage": coverage,
        "abst_precision": float(p),
        "abst_recall": float(r),
        "abst_f1": float(f),
    }
