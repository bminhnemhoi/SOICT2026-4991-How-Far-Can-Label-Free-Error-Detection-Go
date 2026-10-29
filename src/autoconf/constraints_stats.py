"""L2a — Stats-based constraint synthesis.

Deterministic, no LLM.  Emits four constraint families from column stats:
  - domain       (categorical with n_unique below a threshold → allowed set)
  - missingness  (column with low but nonzero missing rate → "rarely null")
  - range        (numerical column → plausible min/max with slack)
  - pattern      (string column whose top values share one character-class shape)
"""

from __future__ import annotations

import re
from typing import Any

from src.autoconf.profiler import DatasetProfile


def synth_stats_constraints(
    profile: DatasetProfile,
    *,
    categorical_unique_threshold: int = 20,
    rare_null_threshold: float = 0.05,
    numerical_range_slack: float = 0.10,
    pattern_min_support: int = 3,
) -> list[dict[str, Any]]:
    """Generate deterministic constraints from profile.column_stats."""
    out: list[dict[str, Any]] = []
    stats = profile.column_stats

    for col, s in stats.items():
        dtype = str(s.get("dtype", ""))
        n_unique = int(s.get("n_unique", 0))
        missing_rate = float(s.get("missing_rate", 0.0))
        is_numeric = "mean" in s and "std" in s and "min" in s and "max" in s

        if 0.0 < missing_rate < rare_null_threshold:
            out.append(_mk_missing(col, missing_rate))

        # categorical domain (only for non-numeric or low-cardinality text cols)
        if not is_numeric and 0 < n_unique <= categorical_unique_threshold:
            top = list(s.get("top_values") or [])
            if top:
                out.append(_mk_domain(col, n_unique, top))

        if is_numeric:
            out.append(_mk_range(col, float(s["min"]), float(s["max"]), numerical_range_slack))

        if not is_numeric:
            top = list(s.get("top_values") or [])
            if len(top) >= pattern_min_support:
                rx = _infer_charclass_regex(top)
                if rx is not None:
                    out.append(_mk_pattern(col, rx, len(top)))

    return out


def _mk_missing(col: str, rate: float) -> dict[str, Any]:
    return {
        "id": f"AUTO-MISS-{col}",
        "type": "missingness",
        "nl": f"Column {col} is rarely null (observed missing rate {rate:.2%}).",
        "formal": {"column": col, "max_missing_rate": round(rate * 1.5, 4)},
        "severity": "low",
        "error_type_when_violated": "missing",
        "source": "auto_stats",
    }


def _mk_domain(col: str, n_unique: int, top: list[Any]) -> dict[str, Any]:
    return {
        "id": f"AUTO-DOM-{col}",
        "type": "domain",
        "nl": f"Column {col} should be one of {n_unique} known values (e.g., {top[:5]}).",
        "formal": {"column": col, "allowed": top},
        "severity": "high" if n_unique <= 5 else "medium",
        "error_type_when_violated": "domain_violation",
        "source": "auto_stats",
    }


def _mk_range(col: str, lo: float, hi: float, slack: float) -> dict[str, Any]:
    span = hi - lo
    pad = max(span * slack, 0.0)
    return {
        "id": f"AUTO-RNG-{col}",
        "type": "range",
        "nl": f"Column {col} within plausible numerical range [{lo - pad:.4g}, {hi + pad:.4g}].",
        "formal": {"column": col, "min": lo - pad, "max": hi + pad},
        "severity": "medium",
        "error_type_when_violated": "outlier",
        "source": "auto_stats",
    }


def _mk_pattern(col: str, regex: str, support: int) -> dict[str, Any]:
    return {
        "id": f"AUTO-PAT-{col}",
        "type": "pattern",
        "nl": f"Column {col} values match character-class pattern {regex}.",
        "formal": {"column": col, "regex": regex},
        "severity": "medium",
        "error_type_when_violated": "pattern",
        "source": "auto_stats",
        "inferred_from_top_n": support,
    }


def _charclass(ch: str) -> str:
    if ch.isdigit():
        return r"\d"
    if ch.isupper():
        return r"[A-Z]"
    if ch.islower():
        return r"[a-z]"
    return re.escape(ch)


def _collapse(tokens: list[str]) -> str:
    """Collapse consecutive identical character-class tokens into {n} quantifier."""
    if not tokens:
        return ""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        j = i
        while j < len(tokens) and tokens[j] == tokens[i]:
            j += 1
        run = j - i
        out.append(tokens[i] if run == 1 else f"{tokens[i]}{{{run}}}")
        i = j
    return "".join(out)


def _abstract(s: Any) -> str | None:
    if s is None:
        return None
    try:
        txt = str(s)
    except Exception:
        return None
    if len(txt) == 0 or len(txt) > 120:
        return None
    return _collapse([_charclass(ch) for ch in txt])


def _infer_charclass_regex(samples: list[Any]) -> str | None:
    """Return a single regex if all samples share the same character-class shape."""
    shapes = set()
    for s in samples:
        shape = _abstract(s)
        if shape is None:
            return None
        shapes.add(shape)
    if len(shapes) == 1:
        return f"^{shapes.pop()}$"
    return None
