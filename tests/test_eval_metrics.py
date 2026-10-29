"""Tests for src/eval/metrics.py"""

from __future__ import annotations

import numpy as np

from src.eval.metrics import (
    abstention_quality,
    aurc,
    brier,
    detection_scores,
    expected_calibration_error,
    reliability_bins,
    risk_at_coverage,
    risk_coverage_curve,
)


def test_detection_perfect_separation():
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    m = detection_scores(y_true, y_score, threshold=0.5)
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["auc_pr"] == 1.0
    assert m["auc_roc"] == 1.0


def test_detection_on_toy(binary_labels_scores):
    y_true, y_score = binary_labels_scores
    m = detection_scores(y_true, y_score)
    assert m["auc_pr"] > y_true.mean()  # beats base rate
    assert m["auc_roc"] > 0.8
    assert 0 <= m["f1"] <= 1


def test_ece_perfect_calibration():
    # Perfectly calibrated: p=0.5 for half positives, half negatives in each bin.
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_conf = np.array([0.5] * 8)
    ece = expected_calibration_error(y_true, y_conf, n_bins=10)
    # mean_conf = 0.5, accuracy = 0.5 → ECE = 0
    assert ece == 0.0


def test_ece_worst_case():
    # Fully wrong: high conf, low accuracy.
    y_true = np.zeros(10)
    y_conf = np.ones(10)
    ece = expected_calibration_error(y_true, y_conf)
    assert ece == 1.0


def test_reliability_bins_shape():
    y_true = np.random.RandomState(0).randint(0, 2, 100)
    y_conf = np.random.RandomState(0).random(100)
    bins = reliability_bins(y_true, y_conf, n_bins=5)
    assert len(bins) == 5
    assert sum(b["n_in_bin"] for b in bins) == 100


def test_brier_basics():
    y_true = np.array([1, 0, 1, 0])
    y_conf = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier(y_true, y_conf) == 0.0
    y_conf_wrong = np.array([0.0, 1.0, 0.0, 1.0])
    assert brier(y_true, y_conf_wrong) == 1.0


def test_risk_coverage_monotone_and_aurc():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.1])
    # Here predictions at threshold 0.5 are [1,1,1,1,0,0] vs true [1,1,0,0,1,0]
    # correct: T,T,F,F,F,T → errors at positions 2,3,4 (indices)
    y_conf = np.array([0.9, 0.8, 0.3, 0.2, 0.4, 0.95])
    curve = risk_coverage_curve(y_true, y_score, y_conf, threshold=0.5)
    assert len(curve) == 101
    assert curve[0]["coverage"] == 0.0
    assert curve[-1]["coverage"] == 1.0
    a = aurc(curve)
    assert 0.0 <= a <= 1.0


def test_risk_at_coverage_interp():
    curve = [{"coverage": 0.0, "risk": 0.0}, {"coverage": 1.0, "risk": 0.5}]
    assert risk_at_coverage(curve, 0.5) == 0.25


def test_abstention_quality_all_flagged():
    y_error = np.array([1, 1, 0, 0])
    abstain = np.array([1, 1, 0, 0])  # perfect
    q = abstention_quality(y_error, abstain)
    assert q["abst_f1"] == 1.0
    assert q["coverage"] == 0.5
