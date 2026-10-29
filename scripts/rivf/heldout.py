"""Held-out generalisation: datasets never inspected while designing the profilers.

Runs the label-free profilers (unchanged hyper-parameters) and RAHA with 20
labelled tuples on RAHA's movies_1 and on the first 20k rows of tax, and
evaluates cell-level P/R/F1 against positional ground truth.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "rivf"))

from run_raha import raha  # noqa: E402  (patched RAHA import)
from src.evidence import profilers as P  # noqa: E402
from src.evidence.fusion import fuse  # noqa: E402

RAW = ROOT / "data" / "raw" / "raha" / "datasets"
OUT = ROOT / "experiments" / "rivf2026" / "heldout"
SPECS = {"movies_1": {"id_cols": {"id"}, "nrows": None}, "tax": {"id_cols": set(), "nrows": 20000}}


def prf(pred, gt):
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    for ds, spec in SPECS.items():
        dirty = pd.read_csv(RAW / ds / "dirty.csv", dtype=str, keep_default_na=False, nrows=spec["nrows"])
        clean = pd.read_csv(RAW / ds / "clean.csv", dtype=str, keep_default_na=False, nrows=spec["nrows"])
        keep = [i for i, c in enumerate(dirty.columns) if c.lower() not in spec["id_cols"]]
        D = dirty.iloc[:, keep].reset_index(drop=True)
        C = clean.iloc[:, keep].reset_index(drop=True)
        gt = D.to_numpy() != C.to_numpy()
        t0 = time.time()
        ev = P.run_profilers(D)
        secs = time.time() - t0
        fz = fuse(ev, P.FAMILY, None, rule="family", use_llm=False)
        R = {"n": len(D), "m": D.shape[1], "cell_err_rate": float(gt.mean()), "row_prev": float(gt.any(1).mean()),
             "profiler_seconds": secs, "profilers": prf(fz["cell"].m_err >= 0.5, gt),
             "per_profiler": {k: prf(E >= 0.5, gt) for k, E in ev.items()},
             "err_by_col": {c: int(gt[:, j].sum()) for j, c in enumerate(D.columns)}}
        # RAHA-20 on the same rows
        tmp = Path(tempfile.mkdtemp(prefix=f"raha_{ds}_"))
        dirty.to_csv(tmp / "dirty.csv", index=False)
        clean.to_csv(tmp / "clean.csv", index=False)
        to_data = {raw: j for j, raw in enumerate(keep)}
        f1s = []
        for seed in (0, 1, 2):
            np.random.seed(seed)
            d = raha.detection.Detection()
            d.LABELING_BUDGET = 20; d.VERBOSE = False; d.SAVE_RESULTS = False
            d.ERROR_DETECTION_ALGORITHMS = ["PVD", "RVD", "KBVD"]
            t1 = time.time()
            det = d.run({"name": ds, "path": str(tmp / "dirty.csv"), "clean_path": str(tmp / "clean.csv")})
            M = np.zeros_like(gt)
            for (i, j) in det.keys():
                if j in to_data:
                    M[i, to_data[j]] = True
            f1s.append({**prf(M, gt), "seconds": time.time() - t1})
            print(ds, "raha seed", seed, f1s[-1], flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
        R["RAHA-20"] = {"F1_mean": float(np.mean([x["F1"] for x in f1s])), "F1_sd": float(np.std([x["F1"] for x in f1s])),
                        "P_mean": float(np.mean([x["P"] for x in f1s])), "R_mean": float(np.mean([x["R"] for x in f1s])),
                        "runs": f1s}
        res[ds] = R
        print(ds, json.dumps({k: v for k, v in R.items() if k != "per_profiler"}, indent=1), flush=True)
        (OUT / "heldout.json").write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
