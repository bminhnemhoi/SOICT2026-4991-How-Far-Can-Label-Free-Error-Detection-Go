"""Re-run RAHA (20 labelled tuples) on the full dirty tables and store cell predictions.

RAHA is run on the complete dataset (as in its paper) and evaluated only on the
cells of the test rows, with positional ground truth.  Several seeds are used
because RAHA samples the tuples to label at random.

Usage: python scripts/rivf/run_raha.py --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if not hasattr(pd.DataFrame, "applymap"):
    pd.DataFrame.applymap = lambda self, func, **kw: self.map(func, **kw)


class _SeqPool:
    def __init__(self, *a, **k): pass
    def map(self, fn, items): return [fn(it) for it in items]
    def close(self): pass
    def join(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): pass


multiprocessing.Pool = _SeqPool
import raha.detection  # noqa: E402

raha.detection.multiprocessing.Pool = _SeqPool

RAW = ROOT / "data" / "raw" / "raha" / "datasets"
OUT = ROOT / "experiments" / "rivf2026" / "raha"


def run(dataset: str, seed: int, budget: int) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix=f"raha_{dataset}_{seed}_"))
    shutil.copy(RAW / dataset / "dirty.csv", tmp / "dirty.csv")
    shutil.copy(RAW / dataset / "clean.csv", tmp / "clean.csv")
    random.seed(seed)
    np.random.seed(seed)
    d = raha.detection.Detection()
    d.LABELING_BUDGET = budget
    d.VERBOSE = False
    d.SAVE_RESULTS = False
    d.ERROR_DETECTION_ALGORITHMS = ["PVD", "RVD", "KBVD"]  # OD's dBoost temp files race on Windows
    t0 = time.time()
    det = d.run({"name": dataset, "path": str(tmp / "dirty.csv"), "clean_path": str(tmp / "clean.csv")})
    el = time.time() - t0
    shutil.rmtree(tmp, ignore_errors=True)
    cells = sorted([int(i), int(j)] for (i, j) in det.keys())
    return {"dataset": dataset, "seed": seed, "budget_tuples": budget, "elapsed_s": el, "cells": cells}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["hospital", "flights", "beers", "rayyan"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--budget", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in args.datasets:
        for s in args.seeds:
            f = OUT / f"{ds}_seed{s}.json"
            if f.exists():
                continue
            res = run(ds, s, args.budget)
            f.write_text(json.dumps(res), encoding="utf-8")
            print(f"{ds} seed={s} flagged={len(res['cells'])} in {res['elapsed_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
