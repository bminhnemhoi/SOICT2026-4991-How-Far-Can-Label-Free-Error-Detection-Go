"""Base LLM agent: load prompt, call LLM, validate output against JSON schema.

Each concrete agent (Auditor, Expert, Judge) subclasses `BaseAgent` and
specifies `prompt_path` and `schema_path`. The base handles:
  - loading prompt markdown (strip YAML front-matter)
  - formatting prompt with INPUT_JSON
  - calling the LLM client (mock or real)
  - parsing + jsonschema validation of output
  - optional self-consistency (M samples at T>0, majority label -> `a`)
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jsonschema

from src.utils.llm_client import LLMClient
from src.utils.io import read_json

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentOutput:
    """Raw + derived signals from an agent call."""

    label: str  # 'err' or 'clean' (mapped from 'error'/'clean')
    p: float  # model confidence in [0, 1]
    c: float  # explanation-evidence consistency in [0, 1]  (default 1.0; NLI later)
    a: float  # self-agreement in [0, 1]  (1.0 if M=1)
    raw: dict[str, Any]  # raw JSON from LLM (schema-validated)
    tokens_in: int
    tokens_out: int
    cache_hit: bool = False
    # Phiên 6 audit pass 3 (C6): list of column names the agent flags as
    # suspicious for the target row. Defaults to [] for back-compat with v1
    # LLM outputs that lack the field.
    suspicious_fields: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "p": self.p,
            "c": self.c,
            "a": self.a,
            "raw": self.raw,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "cache_hit": self.cache_hit,
            "suspicious_fields": list(self.suspicious_fields),
        }


def _strip_front_matter(md: str) -> str:
    """Strip a leading YAML front-matter block delimited by '---' lines."""
    if not md.startswith("---"):
        return md
    lines = md.splitlines()
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1 :]).lstrip()
    return md  # no closing delimiter, keep as-is


def _label_to_internal(label: str) -> str:
    """Map prompt label 'error'/'clean' to BCL internal 'err'/'clean'."""
    if label == "error":
        return "err"
    if label == "clean":
        return "clean"
    raise ValueError(f"Unknown label from LLM: {label!r}")


def load_prompt_template(path: str | Path) -> str:
    raw = Path(path).read_text(encoding="utf-8")
    return _strip_front_matter(raw)


def load_schema(path: str | Path) -> dict[str, Any]:
    return read_json(path)


class BaseAgent:
    """Shared logic for Auditor / Expert / Judge."""

    role: str = "base"
    prompt_path: str = ""
    schema_path: str = ""

    def __init__(
        self,
        client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        self_consistency_m: int = 1,
        self_consistency_temperature: float = 0.7,
        prompt_path: str | None = None,
        schema_path: str | None = None,
    ) -> None:
        self.client = client
        self.model = model or os.environ.get("MA_LED_DEFAULT_MODEL", "gpt-4o-mini-2024-07-18")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed if seed is not None else int(os.environ.get("MA_LED_DEFAULT_SEED", "42"))
        self.self_consistency_m = max(1, int(self_consistency_m))
        self.self_consistency_temperature = self_consistency_temperature
        self.prompt_tmpl = load_prompt_template(prompt_path or self.prompt_path)
        self.schema = load_schema(schema_path or self.schema_path)

    # -- customizable per agent --
    def format_messages(self, payload: dict[str, Any]) -> list[dict[str, str]]:
        """Default: replace {{INPUT_JSON}} with payload as a JSON string."""
        body = self.prompt_tmpl
        if "{{INPUT_JSON}}" in body:
            body = body.replace("{{INPUT_JSON}}", json.dumps(payload, ensure_ascii=False, indent=2))
        else:
            body = body + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)
        return [{"role": "user", "content": body}]

    # -- core call --
    def _strip_unknown_keys(self, parsed: Any, schema: Any) -> Any:
        """Recursively remove keys not in `properties` for objects with
        `additionalProperties: False`. Tolerates LLM-emitted extra fields
        (e.g. `notes`) without weakening downstream consumer contracts.
        """
        if not isinstance(schema, dict):
            return parsed
        stype = schema.get("type")
        if stype == "object" and isinstance(parsed, dict):
            props = schema.get("properties", {}) or {}
            if schema.get("additionalProperties") is False:
                for k in [k for k in parsed.keys() if k not in props]:
                    parsed.pop(k, None)
            for k, sub_schema in props.items():
                if k in parsed:
                    parsed[k] = self._strip_unknown_keys(parsed[k], sub_schema)
        elif stype == "array" and isinstance(parsed, list):
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                parsed = [self._strip_unknown_keys(it, item_schema) for it in parsed]
        return parsed

    def _coerce_top_level_strings(self, parsed: dict[str, Any]) -> dict[str, Any]:
        """Trim top-level string fields to schema maxLength when present.

        Some models occasionally return verbose text that exceeds schema limits
        (e.g., `explanation`, `rationale`). Coercing prevents unnecessary hard
        failures while preserving the main semantic content.
        """
        if not isinstance(parsed, dict):
            return parsed
        props = self.schema.get("properties", {}) if isinstance(self.schema, dict) else {}
        for key, spec in props.items():
            if (
                isinstance(spec, dict)
                and spec.get("type") == "string"
                and isinstance(spec.get("maxLength"), int)
                and isinstance(parsed.get(key), str)
            ):
                max_len = int(spec["maxLength"])
                if len(parsed[key]) > max_len:
                    parsed[key] = parsed[key][:max_len]
        return parsed

    def _call_once(self, messages: list[dict[str, str]], *, temperature: float, seed: int | None) -> dict[str, Any]:
        msgs = list(messages)
        last_err: Exception | None = None
        tokens_in_total = 0
        tokens_out_total = 0
        cache_hit_any = False
        for attempt in range(3):
            resp = self.client.chat(
                msgs,
                model=self.model,
                temperature=temperature,
                max_tokens=self.max_tokens,
                seed=(seed + attempt) if seed is not None else None,
                response_format_json=True,
            )
            tokens_in_total += resp.get("tokens_in", 0)
            tokens_out_total += resp.get("tokens_out", 0)
            cache_hit_any = cache_hit_any or resp.get("cache_hit", False)
            content = resp["content"].strip()
            if not content:
                last_err = ValueError("empty content from LLM (likely reasoning-token exhaustion)")
                msgs = list(messages) + [
                    {"role": "user", "content": (
                        "Your previous response was empty. Return ONLY a single JSON object "
                        "matching the schema. Keep it short. Do not add prose."
                    )},
                ]
                continue
            try:
                parsed = json.loads(content)
                parsed = self._strip_unknown_keys(parsed, self.schema)
                parsed = self._coerce_top_level_strings(parsed)
                jsonschema.validate(parsed, self.schema)
                return {
                    "parsed": parsed,
                    "tokens_in": tokens_in_total,
                    "tokens_out": tokens_out_total,
                    "cache_hit": cache_hit_any,
                }
            except (json.JSONDecodeError, jsonschema.ValidationError) as e:
                last_err = e
                # Append corrective message and retry once. Be schema-agnostic so
                # this works for Auditor/Expert (label) and Judge (decision_hint).
                schema_hint = json.dumps(self.schema, ensure_ascii=False)[:600]
                msgs = list(messages) + [
                    {"role": "assistant", "content": content[:500]},
                    {"role": "user", "content": (
                        f"Your previous response failed schema validation: {str(e)[:200]}. "
                        "Return ONLY valid JSON that conforms strictly to this schema:\n"
                        f"{schema_hint}"
                    )},
                ]
        raise ValueError(f"{self.role}: schema validation failed after retry: {last_err}")

    def run(self, payload: dict[str, Any]) -> AgentOutput:
        """Run agent once + optional self-consistency samples → AgentOutput."""
        messages = self.format_messages(payload)

        # Primary call (deterministic).
        primary = self._call_once(messages, temperature=self.temperature, seed=self.seed)
        parsed = primary["parsed"]
        tokens_in = primary["tokens_in"]
        tokens_out = primary["tokens_out"]
        cache_hit = primary["cache_hit"]

        # Label + confidence (both 'error'/'clean' schemas have these).
        prompt_label = parsed.get("label")
        internal_label = _label_to_internal(prompt_label) if prompt_label in ("error", "clean") else "clean"
        p = float(parsed.get("confidence", 0.5))

        # Self-consistency: M extra samples at T>0, compute agreement ratio `a`.
        a = 1.0
        if self.self_consistency_m > 1:
            labels = [prompt_label]
            for k in range(1, self.self_consistency_m):
                r = self._call_once(
                    messages,
                    temperature=self.self_consistency_temperature,
                    seed=self.seed + k,
                )
                labels.append(r["parsed"].get("label"))
                tokens_in += r["tokens_in"]
                tokens_out += r["tokens_out"]
            most_common, count = Counter(labels).most_common(1)[0]
            a = count / len(labels)
            # Use majority label.
            if most_common in ("error", "clean"):
                internal_label = _label_to_internal(most_common)

        # NLI consistency `c`: stub to 1.0 (real NLI model in Week 2).
        c = self.compute_consistency(parsed, payload)

        # Suspicious fields (C6): optional list of column names. Backward-
        # compatible — absent in pre-C6 LLM outputs and in mock fixtures.
        raw_sf = parsed.get("suspicious_fields", [])
        if isinstance(raw_sf, list):
            sf = tuple(str(x) for x in raw_sf if isinstance(x, (str, int, float)))
        else:
            sf = ()

        return AgentOutput(
            label=internal_label,
            p=max(0.0, min(1.0, p)),
            c=max(0.0, min(1.0, c)),
            a=max(0.0, min(1.0, a)),
            raw=parsed,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cache_hit=cache_hit,
            suspicious_fields=sf,
        )

    # -- overridable: NLI consistency stub --
    def compute_consistency(self, parsed: dict[str, Any], payload: dict[str, Any]) -> float:
        """Lexical proxy for explanation-evidence consistency `c` in [0, 1].

        Heuristic (no NLI model required at this stage):
        - 0.5 if explanation missing/very short (<5 chars)
        - Score = base 0.5 + 0.1 per concrete evidence reference (column name or
          value from payload found in explanation), capped at 1.0
        - This gives the BCL `c` signal real variation instead of constant 1.0
          while we evaluate whether a heavyweight NLI model is needed.

        TODO: replace with mDeBERTa-v3-base-mnli-xnli for the camera-ready run.
        """
        expl = parsed.get("explanation", "")
        if not isinstance(expl, str) or len(expl.strip()) < 5:
            return 0.5

        expl_lc = expl.lower()
        # Collect candidate evidence tokens from payload (column names + cell values).
        evidence_tokens: set[str] = set()
        row_data = payload.get("row") or payload.get("row_data") or {}
        if isinstance(row_data, dict):
            for col, val in row_data.items():
                if isinstance(col, str) and len(col) >= 3:
                    evidence_tokens.add(col.lower())
                if val is not None:
                    sval = str(val).strip().lower()
                    if 3 <= len(sval) <= 60:
                        evidence_tokens.add(sval)
        # Also references to constraints (Expert payload).
        for c_item in payload.get("constraints", []) or []:
            if isinstance(c_item, dict):
                cid = c_item.get("id") or c_item.get("name")
                if isinstance(cid, str) and len(cid) >= 3:
                    evidence_tokens.add(cid.lower())

        if not evidence_tokens:
            return 0.7  # have explanation but no evidence to check against

        hits = sum(1 for tok in evidence_tokens if tok in expl_lc)
        score = 0.5 + 0.1 * hits
        return float(min(1.0, score))
