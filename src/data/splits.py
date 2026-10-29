"""Deterministic train / calibration / test splits.

Used by:
  - BCL Platt scaling fit (calibration split)
  - CADP conformal threshold (calibration split)
  - Final eval (test split)
Split ratios default: 40 / 30 / 30 (train / cal / test).
Deterministic on row index given seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    cal: np.ndarray
    test: np.ndarray

    def as_dict(self) -> dict[str, list[int]]:
        return {"train": self.train.tolist(), "cal": self.cal.tolist(), "test": self.test.tolist()}


def make_splits(
    n: int,
    *,
    train_frac: float = 0.4,
    cal_frac: float = 0.3,
    seed: int = 42,
) -> SplitIndices:
    """Return disjoint index arrays for train / cal / test.

    Args:
        n: total number of rows.
        train_frac, cal_frac: fractions in [0, 1); test_frac = 1 - train - cal.
        seed: RNG seed for deterministic shuffle.
    """
    if train_frac + cal_frac >= 1.0:
        raise ValueError(f"train+cal must be < 1, got {train_frac + cal_frac}")
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * train_frac))
    n_cal = int(round(n * cal_frac))
    train = np.sort(idx[:n_train])
    cal = np.sort(idx[n_train : n_train + n_cal])
    test = np.sort(idx[n_train + n_cal :])
    return SplitIndices(train=train, cal=cal, test=test)
