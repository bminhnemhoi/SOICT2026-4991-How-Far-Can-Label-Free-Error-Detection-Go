"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def binary_labels_scores():
    """Toy binary detection task: 100 samples, clear signal + some noise."""
    rng = np.random.default_rng(42)
    y_true = np.concatenate([np.ones(40), np.zeros(60)]).astype(int)
    # Scores for positives are higher but overlap.
    scores_pos = rng.normal(0.75, 0.15, 40).clip(0, 1)
    scores_neg = rng.normal(0.35, 0.15, 60).clip(0, 1)
    y_score = np.concatenate([scores_pos, scores_neg])
    return y_true, y_score
