"""Checks for individual sentences of the paper (printed and saved to JSON)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evidence import profilers as P  # noqa: E402
from src.evidence.ground_truth import load_benchmark  # noqa: E402

out = {}
# 1) Hospital: share of erroneous cells that are same-length character substitutions
b = load_benchmark("hospital")
clean = pd.read_csv(ROOT / "data/raw/raha/datasets/hospital/clean.csv", dtype=str, keep_default_na=False).iloc[:, b.raw_col_index].to_numpy()
D = b.dirty.to_numpy()
subs = x_only = 0
cells = list(zip(*np.nonzero(b.errors)))
for i, j in cells:
    ops = Levenshtein.editops(clean[i, j], D[i, j])
    if all(t == "replace" for t, _, _ in ops):
        subs += 1
        if all(D[i, j][q] == "x" for _, _, q in ops):
            x_only += 1
out["hospital_substitution_share"] = subs / len(cells)
out["hospital_x_substitution_share"] = x_only / len(cells)

# 2) Hospital sensitivity values
res = json.loads((ROOT / "experiments/rivf2026/results.json").read_text())
out["hospital_sensitivity"] = res["hospital"]["sensitivity"]

# 3) Movies and Rayyan: which profiler produces the false flags (cell evidence >= 0.5)
for name in ("movies_1", "rayyan"):
    d = pd.read_csv(ROOT / f"data/raw/raha/datasets/{name}/dirty.csv", dtype=str, keep_default_na=False)
    c = pd.read_csv(ROOT / f"data/raw/raha/datasets/{name}/clean.csv", dtype=str, keep_default_na=False)
    keep = [k for k, col in enumerate(d.columns) if col.lower() != "id"]
    Dd, Cc = d.iloc[:, keep].reset_index(drop=True), c.iloc[:, keep].to_numpy()
    err = Dd.to_numpy() != Cc
    ev = P.run_profilers(Dd)
    fp_total = int(((np.maximum.reduce(list(ev.values())) >= 0.5) & ~err).sum())
    out[f"{name}_false_flags"] = {"total": fp_total,
                                  "by_profiler": {k: int(((E >= 0.5) & ~err).sum()) for k, E in ev.items()}}
print(json.dumps(out, indent=1))
(ROOT / "experiments/rivf2026/claim_checks.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
