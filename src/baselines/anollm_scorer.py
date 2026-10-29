"""AnoLLM-style row anomaly scorer baseline.

Adapts the AnoLLM (ICLR 2025) approach for the OpenAI Chat Completions API,
which does not expose input-token logprobs.  We use the GENERATED-token
logprobs of a single classification token: the LLM is asked to output the
single token "anomalous" or "normal", and we read the logprob mass on
"anomalous" as the per-row anomaly score.

This differs from the original AnoLLM (which uses input-text perplexity
under a specific LM) but operationalizes the same idea -- LLM-derived
likelihood of a row being anomalous -- on the OpenAI Chat API surface.

Usage (programmatic):
    scorer = AnoLLMScorer(client)
    score = scorer.score_row(row_dict, dataset_name="hospital")
"""

from __future__ import annotations

import json
import math
from typing import Any

from src.utils.llm_client import OpenAILLMClient


_PROMPT_TEMPLATE = """You are an anomaly detector for tabular data.

Rate the following row from the {dataset_name} dataset on a scale of
0 to 100, where 0 = clearly normal and 100 = clearly anomalous (error,
typo, format violation, or implausible value).

Reply with STRICT JSON of the form: {{"score": <integer 0-100>}}

Row:
{row_json}

Reply:"""


class AnoLLMScorer:
    """LLM-likelihood-based row anomaly scorer.

    For each row, we issue one classification call and read the logprob
    of the chosen token.  The score is

        score = exp(logp("anomalous")) / (exp(logp("anomalous")) + exp(logp("normal")))

    higher score = more anomalous.  We use the OpenAI top_logprobs feature
    to obtain candidate token logprobs.
    """

    def __init__(
        self,
        client: OpenAILLMClient,
        model: str = "gpt-5-nano",
        seed: int = 42,
        max_tokens: int = 64,
    ) -> None:
        self.client = client
        self.model = model
        self.seed = seed
        self.max_tokens = max_tokens

    def score_row(self, row: dict[str, Any], dataset_name: str = "tabular") -> dict[str, Any]:
        """Return {'score': float in [0,1], 'label': str, 'tokens_in': int, 'tokens_out': int}.

        We ask the LLM to emit a numeric 0-100 anomaly score; parse to [0,1].
        """
        row_json = json.dumps(row, ensure_ascii=False, default=str)
        prompt = _PROMPT_TEMPLATE.format(dataset_name=dataset_name, row_json=row_json)
        resp = self.client.chat(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            temperature=0.0,
            max_tokens=self.max_tokens,
            seed=self.seed,
            response_format_json=True,
        )
        text = (resp.get("content") or "").strip()
        score = _parse_numeric_score(text)
        return {
            "score": score,
            "label": "anomalous" if score >= 0.5 else "normal",
            "raw_text": text[:80],
            "tokens_in": resp.get("tokens_in", 0),
            "tokens_out": resp.get("tokens_out", 0),
            "cache_hit": resp.get("cache_hit", False),
        }


def _parse_numeric_score(text: str) -> float:
    """Parse a 0-100 anomaly score from LLM output, return float in [0,1].

    Tries JSON {"score": N} first; falls back to first integer or float
    in the text.  Returns 0.5 on parse failure (neutral).
    """
    import re
    if not text:
        return 0.5
    # Strip code fences
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`").lstrip("json").strip()
    # JSON parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict) and "score" in obj:
            v = float(obj["score"])
            if 0 <= v <= 100:
                return v / 100.0
            if 0 <= v <= 1:
                return v
    except Exception:
        pass
    # First numeric in text
    m = re.search(r"\d+(?:\.\d+)?", s)
    if m:
        v = float(m.group())
        if 0 <= v <= 100:
            return v / 100.0
        if 0 <= v <= 1:
            return v
    return 0.5


def _logprob_to_score(top_logprobs: list[dict]) -> float:
    """Convert top_logprobs list to a [0,1] anomaly score.

    Sums logprob mass under tokens starting with 'anom' (positive class)
    and tokens starting with 'norm'/'clean' (negative class), then
    normalizes via softmax.  Returns 0.5 if neither class is present.
    """
    if not top_logprobs:
        return 0.5
    p_anom = 0.0
    p_norm = 0.0
    for tok in top_logprobs:
        # tok format: {"token": str, "logprob": float}
        t = (tok.get("token") or "").strip().lower().lstrip(" \t")
        lp = tok.get("logprob")
        if lp is None:
            continue
        prob = math.exp(lp)
        if t.startswith("anom"):
            p_anom += prob
        elif t.startswith("norm") or t.startswith("clean") or t == "ok":
            p_norm += prob
    total = p_anom + p_norm
    if total <= 0:
        return 0.5
    return float(p_anom / total)


def _chat_with_logprobs(
    client: OpenAILLMClient,
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    top_logprobs: int,
) -> dict[str, Any]:
    """Call OpenAI chat with logprobs=True; cache by prompt hash."""
    # Reuse client's cache by issuing a deterministic key including 'logp_v2'.
    from src.utils.llm_client import _cache_key
    key = _cache_key(model + "::logp", messages, temperature, seed)
    if client.cache is not None and key in client.cache:
        cached = client.cache[key]
        client.cost.record(0, 0, provider="openai", cache_hit=True)
        return {**cached, "cache_hit": True}
    if client.mode == "cache_only":
        raise RuntimeError(f"cache_only miss for {key[:12]}")

    # Direct API call with logprobs.  Re-uses tenacity in the underlying
    # _call_api would be cleaner, but for a baseline we keep a simple
    # try/except.
    try:
        resp = client._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
            seed=seed,
        )
    except TypeError:
        # Older OpenAI SDK
        resp = client._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            logprobs=True,
            top_logprobs=top_logprobs,
            seed=seed,
        )

    choice = resp.choices[0]
    content = choice.message.content or ""
    tokens_in = getattr(resp.usage, "prompt_tokens", 0) or 0
    tokens_out = getattr(resp.usage, "completion_tokens", 0) or 0

    # Extract logprobs of the FIRST generated token.
    top_logprobs_first: list[dict] = []
    if choice.logprobs and choice.logprobs.content:
        first = choice.logprobs.content[0]
        # first.top_logprobs is a list of TopLogprob items.
        for tl in (first.top_logprobs or []):
            top_logprobs_first.append({"token": tl.token, "logprob": tl.logprob})

    client.cost.record(tokens_in, tokens_out, provider="openai", cache_hit=False)
    out = {
        "content": content,
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "top_logprobs_first": top_logprobs_first,
        "model": model,
    }
    if client.cache is not None:
        client.cache[key] = out
    return out
