"""Beers/all tables: cell F1 when P5 keeps only its first branch (majority residue is the convention)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evidence import profilers as P  # noqa: E402
from src.evidence.fusion import fuse  # noqa: E402
from src.evidence.ground_truth import load_benchmark  # noqa: E402


def unit_first_branch(df: pd.DataFrame) -> np.ndarray:
    n, m = df.shape
    E = np.zeros((n, m))
    for j, c in enumerate(df.columns):
        s = df[c]
        nonnull = ~s.str.strip().str.lower().isin(P.NULL_TOKENS)
        single = s.map(lambda v: len(P._NUM.findall(v)) == 1) & nonnull
        if nonnull.sum() == 0 or single.sum() < P.NUMERIC_MIN_SHARE * nonnull.sum():
            continue
        residue = s[single].map(lambda v: P._NUM.sub("#", v, count=1).strip())
        if residue.str.startswith("#").mean() < P.UNIT_SUFFIX_SHARE:
            continue
        cnt = residue.value_counts(normalize=True)
        if len(cnt) > P.UNIT_MAX_FORMATS or len(cnt) == 1:
            continue
        top = cnt.index[0]  # majority residue is always the convention
        strength = 1.0 - residue.map(cnt)
        bad = residue != top
        idx = np.flatnonzero(single.to_numpy())
        E[idx[bad.to_numpy()], j] = strength[bad].to_numpy()
    return E


out = {}
for ds in ("hospital", "flights", "beers", "rayyan"):
    b = load_benchmark(ds)
    t = b.splits["test"]
    ev = P.run_profilers(b.dirty)
    ev["unit"] = unit_first_branch(b.dirty)
    flag = fuse(ev, P.FAMILY, None, rule="family", use_llm=False)["cell"].m_err[t] >= 0.5
    gt = b.errors[t]
    tp = (flag & gt).sum(); fp = (flag & ~gt).sum(); fn = (~flag & gt).sum()
    out[ds] = float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0
print(out)
(ROOT / "experiments" / "rivf2026" / "p5_first_branch.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
