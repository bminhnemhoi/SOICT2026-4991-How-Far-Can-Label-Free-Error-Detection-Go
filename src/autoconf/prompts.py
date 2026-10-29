"""L3 - Auto-generate Auditor / Expert prompt templates from a DatasetProfile.

Goal: drop the per-dataset hand-tuned prompt files (prompts/expert_v1.md etc.)
in favour of a templater that takes (profile, constraints) -> rendered prompt.

For now we generate ONE template (the Expert prompt); the Auditor prompt is
already domain-agnostic and uses column_stats only.

The optional `infer_domain` helper does a single low-token LLM call to name
the domain from column headers + sample rows; this is what makes the prompt
"feel" like a hand-crafted expert prompt without manual editing.
"""

from __future__ import annotations

import json
from typing import Any

from src.autoconf.profiler import DatasetProfile
from src.utils.llm_client import LLMClient


_DOMAIN_INFER_PROMPT = """Given the following column headers and sample rows, name the domain in 2-5 words (English).  Output JSON: {{"domain": "<name>"}}.

# Columns
{cols}

# Sample rows
{sample}
"""


def infer_domain(
    profile: DatasetProfile,
    client: LLMClient,
    *,
    model: str = "gpt-5-nano",
) -> str:
    """One low-cost LLM call: 'Healthcare hospital quality reports', etc."""
    sample = json.dumps(profile.sample_rows[:5], default=str)[:1500]
    prompt = _DOMAIN_INFER_PROMPT.format(
        cols=", ".join(profile.columns),
        sample=sample,
    )
    try:
        resp = client.chat(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=0.0,
            max_tokens=100,
            seed=42,
            response_format_json=True,
        )
        content = (resp.get("content") or "").strip()
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "domain" in parsed:
            return str(parsed["domain"])[:80]
    except Exception:
        pass
    return "tabular data"


_EXPERT_TEMPLATE = """---
role: expert
version: auto-1
generated_from: profile({name})
domain: {domain}
---

# System

You are a domain expert on **{domain}**.  Your task: given ONE row from this dataset, decide whether it contains a data-quality error, using the constraints below + cross-row neighbour rows when provided.  You may use commonsense world-knowledge about the domain.

# Schema

The dataset has columns: {col_list}.

# Constraints (priority HIGH > MEDIUM > LOW)

{constraints_block}

# Input

You receive a JSON object with:
- `row_id`: int
- `target` (optional): {{"column": "...", "value": ...}} - focus on this cell if set
- `row`: dict of column -> value
- `neighbors` (optional): list of nearby rows, for cross-row reasoning
- `constraints`: subset of constraints above selected as relevant for this row

# Reasoning (do this internally; output ONLY JSON)

1. For each HIGH-severity constraint that references a column in this row, check whether the row violates it.
2. For each MEDIUM constraint, check format / pattern / allowed values.
3. For functional dependencies, compare against neighbour rows (if provided): if the same LHS value appears with a different RHS value among neighbours, that's a violation.
4. If 0 violations -> label "clean", confidence 0.7-0.85.
5. If 1+ violations -> label "error", confidence 0.75-0.95 depending on severity.

# Output (STRICT JSON, no prose)

{{
  "label": "error" | "clean",
  "confidence": <0..1>,
  "explanation": "<= 80 words; cite the constraint id(s) you used>",
  "suspicious_fields": ["<col>", ...],
  "evidence_features": {{
    "violated_constraints": ["<constraint_id>", ...],
    "fd_neighbour_conflicts": ["<short str>", ...]
  }}
}}

INPUT:
{{{{INPUT_JSON}}}}
"""


def _format_constraint_line(c: dict[str, Any]) -> str:
    sev = str(c.get("severity", "medium")).upper()
    cid = c.get("id", "?")
    nl = c.get("nl", "")
    return f"  - [{sev:6s}] {cid}: {nl}"


def build_expert_prompt(
    profile: DatasetProfile,
    constraints: list[dict[str, Any]],
    *,
    domain_hint: str | None = None,
    max_constraints_in_prompt: int = 20,
) -> str:
    """Render a complete Expert prompt for the MA-LED pipeline."""
    domain = domain_hint or profile.inferred_domain or "tabular data"
    sorted_constraints = sorted(
        constraints,
        key=lambda c: {"high": 0, "medium": 1, "low": 2}.get(
            str(c.get("severity", "medium")).lower(), 1
        ),
    )[:max_constraints_in_prompt]

    constraints_block = (
        "\n".join(_format_constraint_line(c) for c in sorted_constraints)
        or "  (no constraints provided)"
    )
    return _EXPERT_TEMPLATE.format(
        name=profile.name,
        domain=domain,
        col_list=", ".join(profile.columns),
        constraints_block=constraints_block,
    )
