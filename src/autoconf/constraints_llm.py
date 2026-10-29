"""L2c - LLM-assisted constraint synthesis.

Single LLM call per dataset.  Given the schema profile + a small sample, asks
the model to propose 5-10 domain constraints in our YAML schema.  Output is
parsed, schema-validated, and tagged with source='auto_llm'.

Cost:  ~1500-3000 input tokens + 500-1500 output -> ~$0.001-0.003 per dataset
       on gpt-5-nano.
"""

from __future__ import annotations

import json
import re
from typing import Any

from src.autoconf.profiler import DatasetProfile
from src.utils.llm_client import LLMClient


_PROMPT = """You are a data-quality expert.  Below is the schema and a sample of a tabular dataset.  Propose between 5 and {max_constraints} domain constraints that dirty rows are likely to violate.  Use the JSON schema given.

# Dataset name
{name}

# Column statistics
{stats_json}

# Sample rows ({k_sample} random)
{sample_json}

# Output schema
Return a JSON object: {{"constraints": [<constraint>, ...]}}

Each constraint:
  {{
    "id": "LLM-<TYPE>-<NNN>",
    "type": "functional_dependency" | "pattern" | "domain" | "business_rule" | "range",
    "nl": "<short natural-language description, <= 30 words>",
    "formal": <object specific to type; see below>,
    "severity": "high" | "medium" | "low",
    "error_type_when_violated": "fd_violation" | "pattern" | "domain_violation" | "business_rule" | "outlier"
  }}

Formal shapes:
  - functional_dependency: {{"lhs": ["col"], "rhs": ["col"]}}
  - pattern:               {{"column": "col", "regex": "^...$"}}
  - domain:                {{"column": "col", "allowed": ["v1", "v2", ...]}}
  - business_rule:         {{"column": "col", "rule": "<NL only>"}}
  - range:                 {{"column": "col", "min": <num>, "max": <num>}}

Rules:
- Use EXACT column names from the schema.
- Prefer functional_dependency and pattern when the data clearly supports it.
- Avoid trivial rules (e.g., "every column should have a value").
- Do not duplicate rules captured by the column stats already.
- Output ONLY the JSON object; no prose.
"""


def synth_llm_constraints(
    profile: DatasetProfile,
    client: LLMClient,
    *,
    model: str = "gpt-5-nano",
    max_constraints: int = 10,
    temperature: float = 0.0,
    max_input_chars: int = 4000,
) -> list[dict[str, Any]]:
    """Query an LLM for domain constraints derived from the profile.

    Returns a list of constraint dicts (empty list on parse / API failure).
    """
    sample_for_prompt = profile.sample_rows[: min(10, len(profile.sample_rows))]
    stats_json = json.dumps(profile.column_stats, indent=2, default=str)[:max_input_chars]
    sample_json = json.dumps(sample_for_prompt, indent=2, default=str)[:max_input_chars]

    prompt = _PROMPT.format(
        name=profile.name,
        stats_json=stats_json,
        sample_json=sample_json,
        k_sample=len(sample_for_prompt),
        max_constraints=max_constraints,
    )

    try:
        resp = client.chat(
            [{"role": "user", "content": prompt}],
            model=model,
            temperature=temperature,
            max_tokens=2500,
            seed=42,
            response_format_json=True,
        )
    except Exception as e:
        print(f"[autoconf.llm] API call failed: {e}")
        return []

    content = (resp.get("content") or "").strip()
    if not content:
        return []

    parsed = _safe_parse(content)
    raw_list = _extract_constraints(parsed)
    if not raw_list:
        return []

    out: list[dict[str, Any]] = []
    valid_columns = set(profile.columns)
    for c in raw_list[:max_constraints]:
        if not isinstance(c, dict):
            continue
        c2 = _sanitize(c, valid_columns)
        if c2 is not None:
            c2.setdefault("source", "auto_llm")
            out.append(c2)
    return out


def _safe_parse(content: str) -> Any:
    if content.startswith("```"):
        m = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", content, flags=re.DOTALL)
        if m:
            content = m.group(1)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


def _extract_constraints(parsed: Any) -> list[Any]:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        cs = parsed.get("constraints")
        if isinstance(cs, list):
            return cs
    return []


_ALLOWED_TYPES = {
    "functional_dependency", "pattern", "domain", "business_rule", "range"
}
_ALLOWED_SEV = {"high", "medium", "low"}


def _sanitize(c: dict[str, Any], valid_cols: set[str]) -> dict[str, Any] | None:
    """Validate and normalize a single LLM-emitted constraint.

    Drops the constraint (returns None) if structure / column refs are bad.
    """
    ctype = str(c.get("type", "")).lower()
    if ctype not in _ALLOWED_TYPES:
        return None

    formal = c.get("formal") or {}
    if not isinstance(formal, dict):
        return None

    referenced: set[str] = set()
    if ctype == "functional_dependency":
        lhs = formal.get("lhs") or []
        rhs = formal.get("rhs") or []
        if not (isinstance(lhs, list) and isinstance(rhs, list) and lhs and rhs):
            return None
        referenced.update(lhs)
        referenced.update(rhs)
    elif ctype in ("pattern", "domain", "business_rule", "range"):
        col = formal.get("column")
        if not isinstance(col, str):
            return None
        referenced.add(col)
        if ctype == "pattern" and not isinstance(formal.get("regex"), str):
            return None
        if ctype == "domain":
            allowed = formal.get("allowed") or []
            if not (isinstance(allowed, list) and allowed):
                return None
        if ctype == "range":
            try:
                float(formal.get("min"))
                float(formal.get("max"))
            except (TypeError, ValueError):
                return None

    if not referenced.issubset(valid_cols):
        return None

    sev = str(c.get("severity", "medium")).lower()
    if sev not in _ALLOWED_SEV:
        sev = "medium"

    return {
        "id": str(c.get("id") or f"LLM-{ctype.upper()[:4]}-AUTO"),
        "type": ctype,
        "nl": str(c.get("nl", ""))[:200],
        "formal": formal,
        "severity": sev,
        "error_type_when_violated": str(
            c.get("error_type_when_violated", "domain_violation")
        ),
    }
