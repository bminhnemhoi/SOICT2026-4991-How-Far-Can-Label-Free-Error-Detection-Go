"""Which annotated errors are only formatting conventions?

Every erroneous cell (dirty != clean) of the six tables is put in one class:
  missing  - exactly one of dirty/clean is an explicit null token
  format   - same numbers (as a multiset, after removing thousands separators)
             or same letters/digits after case-folding: the value is right but
             written in another convention (units, separators, trailing .0,
             leading zeros, date order, case)
  content  - anything else (typos, wrong or conflicting values)
For format errors we also record whether the dirty value has the column's
most frequent character shape, i.e. whether the dirty table is internally
consistent, which no table-internal detector can contradict.
Profiler recall is reported per class on the whole table.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evidence import profilers as P  # noqa: E402
from src.evidence.fusion import fuse  # noqa: E402
from src.evidence.ground_truth import load_benchmark  # noqa: E402

RAW = ROOT / "data" / "raw" / "raha" / "datasets"
SEP = re.compile(r"(?<=\d),(?=\d{3}\b)")
NUM = re.compile(r"\d+(?:\.\d+)?")
ALNUM = re.compile(r"[^0-9a-z]")
LETTERS = re.compile(r"[^a-z]")


def is_null(v: str) -> bool:
    return v.strip().lower() in P.NULL_TOKENS


def classify(d: str, c: str) -> str:
    if is_null(d) != is_null(c):
        return "missing"
    if is_null(d) and is_null(c):
        return "format"
    nd = sorted(float(x) for x in NUM.findall(SEP.sub("", d)))
    nc = sorted(float(x) for x in NUM.findall(SEP.sub("", c)))
    ld, lc = LETTERS.sub("", d.lower()), LETTERS.sub("", c.lower())
    # same numbers and the same words, or words (units) present on one side only
    if nd and nd == nc and (ld == lc or not ld or not lc):
        return "format"
    if ALNUM.sub("", d.lower()) == ALNUM.sub("", c.lower()):
        return "format"
    return "content"


def load(name: str):
    if name in ("hospital", "flights", "beers", "rayyan"):
        b = load_benchmark(name)
        clean_raw = pd.read_csv(RAW / name / "clean.csv", dtype=str, keep_default_na=False)
        C = clean_raw.iloc[:, b.raw_col_index].to_numpy()
        return b.dirty, C
    nrows = 20000 if name == "tax" else None
    d = pd.read_csv(RAW / name / "dirty.csv", dtype=str, keep_default_na=False, nrows=nrows)
    c = pd.read_csv(RAW / name / "clean.csv", dtype=str, keep_default_na=False, nrows=nrows)
    keep = [i for i, col in enumerate(d.columns) if col.lower() != "id"]
    D = d.iloc[:, keep].reset_index(drop=True)
    return D, c.iloc[:, keep].to_numpy()


def main() -> None:
    out = {}
    for name in ("hospital", "flights", "beers", "rayyan", "movies_1", "tax"):
        D, C = load(name)
        Dv = D.to_numpy()
        err = Dv != C
        modal = {j: Counter(D.iloc[:, j].map(P.shape)).most_common(1)[0][0] for j in range(D.shape[1])}
        flags = fuse(P.run_profilers(D), P.FAMILY, None, rule="family", use_llm=False)["cell"].m_err >= 0.5
        cls = Counter(); caught = Counter(); fmt_modal = 0
        for i, j in zip(*np.nonzero(err)):
            k = classify(Dv[i, j], C[i, j])
            cls[k] += 1
            caught[k] += int(flags[i, j])
            if k == "format" and P.shape(Dv[i, j]) == modal[j]:
                fmt_modal += 1
        n_err = int(err.sum())
        tp = int((flags & err).sum()); fp = int((flags & ~err).sum()); fn = int((~flags & err).sum())
        prof_f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        raha_f1 = None
        if name in ("hospital", "flights", "beers", "rayyan"):
            raw_idx = load_benchmark(name).raw_col_index
            to_data = {raw: j for j, raw in enumerate(raw_idx)}
            vals = []
            for f in sorted((ROOT / "experiments" / "rivf2026" / "raha").glob(f"{name}_seed*.json")):
                M = np.zeros_like(err)
                for i, j in json.loads(f.read_text())["cells"]:
                    if j in to_data:
                        M[i, to_data[j]] = True
                t2 = int((M & err).sum()); f2 = int((M & ~err).sum()); n2 = int((~M & err).sum())
                vals.append(2 * t2 / (2 * t2 + f2 + n2))
            raha_f1 = [float(np.mean(vals)), float(np.std(vals))]
        else:
            ho = ROOT / "experiments" / "rivf2026" / "heldout" / "heldout.json"
            if ho.exists():
                h = json.loads(ho.read_text()).get(name)
                if h:
                    raha_f1 = [h["RAHA-20"]["F1_mean"], h["RAHA-20"]["F1_sd"]]
        out[name] = {
            "rows": int(D.shape[0]), "cols": int(D.shape[1]), "cell_err_rate": float(err.mean()),
            "row_prev": float(err.any(1).mean()),
            "profiler_F1_whole": prof_f1, "RAHA20_F1_whole": raha_f1,
            "n_err": n_err,
            "share": {k: cls[k] / n_err for k in ("content", "missing", "format")},
            "profiler_recall": {k: (caught[k] / cls[k] if cls[k] else None) for k in ("content", "missing", "format")},
            "format_with_modal_shape": fmt_modal / cls["format"] if cls["format"] else None,
        }
        print(name, json.dumps(out[name]), flush=True)
    (ROOT / "experiments" / "rivf2026" / "error_taxonomy.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
