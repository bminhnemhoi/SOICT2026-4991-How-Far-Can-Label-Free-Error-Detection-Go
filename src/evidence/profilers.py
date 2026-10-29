"""Label-free, configuration-free data profilers that emit cell-level error evidence.

Every profiler maps the dirty table D (n rows x m columns, string values) to an
evidence matrix E in [0,1]^{n x m}; E[i,j] > 0 means "cell (i,j) looks
erroneous" with strength E[i,j].  A zero entry means *no opinion* (open-world),
not "clean".  All hyper-parameters are fixed module constants chosen on the dev
split and applied unchanged to every dataset.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

# --------------------------------------------------------------------------
# Fixed hyper-parameters (never tuned on test rows)
# --------------------------------------------------------------------------
PATTERN_RARE_SHARE = 0.05     # a shape covering < 5% of a column is "rare"
TYPO_MIN_RATIO = 90.0         # rapidfuzz ratio for a near-duplicate (dev-selected)
TYPO_DOMINANCE = 3            # frequent value must be >= 3x more frequent
FD_MIN_CONF = 0.90            # majority confidence for an approximate FD
FD_MAX_LHS = 1                # single-column determinants
FD_MIN_GROUP = 3              # group must have >= 3 rows to vote
NULL_TOKENS = {"", "n/a", "na", "null", "none", "nan", "-", "--", "?", "#n/a"}

_UP = re.compile(r"[A-Z]+")
_LO = re.compile(r"[a-z]+")
_DG = re.compile(r"[0-9]+")
_WS = re.compile(r"\s+")


def shape(v: str) -> str:
    """Character-class shape: 'Birmingham 35235' -> 'Aa 9'."""
    s = _UP.sub("A", v)
    s = _LO.sub("a", s)
    s = _DG.sub("9", s)
    return _WS.sub(" ", s)


# --------------------------------------------------------------------------
# P1: syntactic pattern rarity
# --------------------------------------------------------------------------
def pattern_rarity(df: pd.DataFrame) -> np.ndarray:
    n, m = df.shape
    E = np.zeros((n, m))
    for j, c in enumerate(df.columns):
        shapes = df[c].map(shape)
        cnt = shapes.value_counts()
        share = shapes.map(cnt).to_numpy() / n
        dominance = cnt.iloc[0] / n  # regularity of the column
        rare = np.clip(1.0 - share / PATTERN_RARE_SHARE, 0.0, 1.0)
        E[:, j] = dominance * rare
    return E


# --------------------------------------------------------------------------
# P2: disguised / explicit missing values
# --------------------------------------------------------------------------
def missing_tokens(df: pd.DataFrame) -> np.ndarray:
    n, m = df.shape
    E = np.zeros((n, m))
    for j, c in enumerate(df.columns):
        low = df[c].str.strip().str.lower()
        is_null = low.isin(NULL_TOKENS).to_numpy()
        frac = is_null.mean()
        if frac == 0.0 or frac > 0.5:
            # a column that is mostly null uses null as its normal value
            continue
        E[is_null, j] = 1.0 - frac
    return E


# --------------------------------------------------------------------------
# P3: near-duplicate typos of a much more frequent value
# --------------------------------------------------------------------------
def _alpha_edit(a: str, b: str) -> bool:
    """True if the edit script between a and b touches an alphabetic character
    (pure digit changes such as 6:55 vs 6:59 are different values, not typos)."""
    for tag, i, j in Levenshtein.editops(a, b):
        ca = a[i] if i < len(a) else ""
        cb = b[j] if j < len(b) else ""
        if ca.isalpha() or cb.isalpha():
            return True
    return False


def near_duplicate_typos(df: pd.DataFrame) -> np.ndarray:
    n, m = df.shape
    E = np.zeros((n, m))
    for j, c in enumerate(df.columns):
        vc = Counter(df[c].tolist())
        frequent = [v for v, k in vc.items() if k >= TYPO_DOMINANCE and len(v) >= 3]
        if not frequent:
            continue
        score: dict[str, float] = {}
        for v, k in vc.items():
            if len(v) < 3:
                continue
            cands = process.extract(v, frequent, scorer=fuzz.ratio, limit=3, score_cutoff=TYPO_MIN_RATIO)
            best = 0.0
            for u, r, _ in cands:
                if u == v or vc[u] < TYPO_DOMINANCE * k:
                    continue
                if not _alpha_edit(v, u):
                    continue
                best = max(best, (r / 100.0) * vc[u] / (vc[u] + k))
            if best > 0:
                score[v] = best
        if score:
            E[:, j] = df[c].map(lambda x: score.get(x, 0.0)).to_numpy()
    return E


# --------------------------------------------------------------------------
# P4: violations of automatically mined approximate functional dependencies
# --------------------------------------------------------------------------
def mine_fds(df: pd.DataFrame) -> list[tuple[str, str, float]]:
    """Approximate FDs X -> Y with |X| = 1 on the dirty table.

    conf(X->Y) = (1/n_g) sum_g |{r in g : Y(r) = mode_g(Y)}| over groups with
    >= FD_MIN_GROUP rows.  Key-like and constant columns are skipped."""
    n = len(df)
    nun = df.nunique()
    out = []
    for x in df.columns:
        if nun[x] <= 1 or nun[x] >= 0.9 * n:
            continue
        sizes = df.groupby(x, sort=False)[x].transform("size")
        big = sizes >= FD_MIN_GROUP
        if big.sum() < 0.3 * n:
            continue
        sub = df[big]
        for y in df.columns:
            if y == x or nun[y] <= 1:
                continue
            # confidence over non-null RHS values (missing cells are P2's job)
            ok = ~sub[y].str.strip().str.lower().isin(NULL_TOKENS)
            if ok.sum() < 0.3 * n:
                continue
            mode_cnt = sub[ok].groupby(x, sort=False)[y].agg(lambda s: s.value_counts().iloc[0])
            conf = mode_cnt.sum() / ok.sum()
            # skip trivial FDs where Y is (almost) constant within the table
            if conf >= FD_MIN_CONF and df[y].value_counts(normalize=True).iloc[0] < 0.9:
                out.append((x, y, float(conf)))
    return out


def fd_violations(df: pd.DataFrame, fds: list[tuple[str, str, float]] | None = None) -> np.ndarray:
    n, m = df.shape
    col_idx = {c: j for j, c in enumerate(df.columns)}
    E = np.zeros((n, m))
    if fds is None:
        fds = mine_fds(df)
    for x, y, conf in fds:
        # vote only among non-null RHS values; null cells are left to P2
        ok = ~df[y].str.strip().str.lower().isin(NULL_TOKENS)
        sub = df.loc[ok, [x, y]]
        g = sub.groupby(x, sort=False)[y]
        size = g.transform("size").to_numpy()
        mode_val = g.transform(lambda s: s.value_counts().index[0]).to_numpy()
        mode_cnt = g.transform(lambda s: s.value_counts().iloc[0]).to_numpy()
        val_cnt = sub.groupby([x, y], sort=False)[y].transform("size").to_numpy()
        yv = sub[y].to_numpy()
        viol = (size >= FD_MIN_GROUP) & (yv != mode_val) & (mode_cnt > val_cnt)
        # evidence grows with the dominance of the group's majority value
        # (conf is used only to select the FD; chosen on the dev split)
        strength = 1.0 - val_cnt / np.maximum(mode_cnt, 1)
        rows = np.flatnonzero(ok.to_numpy())[viol]
        j = col_idx[y]
        E[rows, j] = np.maximum(E[rows, j], strength[viol])
    return E


# --------------------------------------------------------------------------
# P5: representational (unit / symbol) inconsistency in numeric columns
# --------------------------------------------------------------------------
_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")
NUMERIC_MIN_SHARE = 0.8       # >= 80% of non-null cells hold exactly one number
UNIT_SUFFIX_SHARE = 0.9       # >= 90% of those write the number first
UNIT_MAX_FORMATS = 10         # a handful of formats, not free text
UNIT_CONSISTENT = 0.9         # one format covering >= 90% is the convention


def unit_inconsistency(df: pd.DataFrame) -> np.ndarray:
    n, m = df.shape
    E = np.zeros((n, m))
    for j, c in enumerate(df.columns):
        s = df[c]
        nonnull = ~s.str.strip().str.lower().isin(NULL_TOKENS)
        single = s.map(lambda v: len(_NUM.findall(v)) == 1) & nonnull
        if nonnull.sum() == 0 or single.sum() < NUMERIC_MIN_SHARE * nonnull.sum():
            continue
        residue = s[single].map(lambda v: _NUM.sub("#", v, count=1).strip())
        if residue.str.startswith("#").mean() < UNIT_SUFFIX_SHARE:
            continue
        cnt = residue.value_counts(normalize=True)
        if len(cnt) > UNIT_MAX_FORMATS or len(cnt) == 1:
            continue
        top, top_share = cnt.index[0], cnt.iloc[0]
        canonical = top if top_share >= UNIT_CONSISTENT else "#"
        strength = 1.0 - residue.map(cnt)
        bad = residue != canonical
        idx = np.flatnonzero(single.to_numpy())
        E[idx[bad.to_numpy()], j] = strength[bad].to_numpy()
    return E


PROFILERS = {
    "pattern": pattern_rarity,
    "missing": missing_tokens,
    "typo": near_duplicate_typos,
    "fd": fd_violations,
    "unit": unit_inconsistency,
}

# Families of non-distinct sources: within a family evidence is combined with
# the cautious rule (max weight), across families with Dempster's rule (sum).
FAMILY = {"pattern": "syntax", "unit": "syntax", "missing": "syntax", "typo": "value", "fd": "value"}


def run_profilers(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {name: fn(df) for name, fn in PROFILERS.items()}
