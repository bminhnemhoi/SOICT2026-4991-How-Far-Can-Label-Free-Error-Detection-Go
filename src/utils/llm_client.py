"""LLM client with cache + retry + mock mode.

3 modes controlled by env `MA_LED_LLM_MODE`:
- real       : call provider API (default for production)
- mock       : deterministic MockLLMClient (dev/test, $0)
- cache_only : fail loud if cache miss (verify reproducibility)

Per docs/12 §3.4. API key is loaded from env, NEVER logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any, Protocol

import diskcache
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils.cost import CostTracker

logger = logging.getLogger(__name__)


def _cache_key(model: str, messages: list[dict], temperature: float, seed: int | None) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "seed": seed,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class LLMClient(Protocol):
    """Minimal interface all backends must implement."""

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        response_format_json: bool = True,
    ) -> dict[str, Any]:
        """Return {'content': str, 'tokens_in': int, 'tokens_out': int, 'cache_hit': bool}."""
        ...


class MockLLMClient:
    """Deterministic mock. Returns canned JSON by prompt hash, or schema-shaped default.

    Default synthesized responses are role-aware: the mock inspects the messages
    for role hints ('Auditor', 'Domain Expert', 'Judge') and emits a JSON object
    matching the corresponding prompt schema (prompts/schemas/*.json).
    """

    def __init__(self, canned: dict[str, str] | None = None) -> None:
        self.canned = canned or {}

    @staticmethod
    def _detect_role(messages: list[dict]) -> str:
        blob = " ".join(m.get("content", "") for m in messages).lower()
        # Order matters: Expert prompt references "Auditor" (and vice versa).
        # Prefer the role whose name appears FIRST in the system/user prompt.
        positions = {
            "expert": blob.find("you are a **domain expert"),
            "auditor": blob.find("you are a **data auditor"),
            "judge": blob.find("you are the **judge"),
        }
        positions = {r: p for r, p in positions.items() if p >= 0}
        if positions:
            return min(positions, key=positions.get)
        # Fallback: any mention
        if "domain expert" in blob:
            return "expert"
        if "data auditor" in blob or "auditor" in blob:
            return "auditor"
        if "judge" in blob:
            return "judge"
        return "generic"

    @staticmethod
    def _default_payload(role: str) -> dict[str, Any]:
        if role == "auditor":
            return {
                "label": "clean",
                "confidence": 0.5,
                "explanation": "[MOCK] no anomalies detected.",
                "evidence_features": {
                    "stat_anomalies": [],
                    "pattern_issues": [],
                    "missingness_issues": [],
                },
            }
        if role == "expert":
            return {
                "label": "clean",
                "confidence": 0.5,
                "explanation": "[MOCK] no constraint violations detected.",
                "evidence_features": {
                    "fd_violations": [],
                    "denial_violations": [],
                    "rule_violations": [],
                    "cross_row_issues": [],
                },
            }
        if role == "judge":
            return {
                "decision_hint": "abstain",
                "error_type": "none",
                "suspicious_fields": [],
                "rationale": "[MOCK] no strong evidence.",
            }
        return {
            "p": 0.5,
            "c": 0.5,
            "evidence": "[MOCK] synthesized response",
            "constraint_id": None,
            "error_type_hint": None,
        }

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        response_format_json: bool = True,
    ) -> dict[str, Any]:
        key = _cache_key(model, messages, temperature, seed)
        if key in self.canned:
            content = self.canned[key]
        else:
            role = self._detect_role(messages)
            content = json.dumps(self._default_payload(role), ensure_ascii=False)
        # Rough token estimate for tests.
        tokens_in = sum(len(m.get("content", "").split()) for m in messages) * 2
        tokens_out = len(content.split()) * 2
        return {
            "content": content,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cache_hit": False,
            "model": f"mock:{model}",
        }


class OpenAILLMClient:
    """OpenAI wrapper with retry + cache. API key from OPENAI_API_KEY env."""

    def __init__(
        self,
        cache_path: str | None = ".cache/llm.sqlite",
        cache_enabled: bool = True,
        cost_tracker: CostTracker | None = None,
        mode: str = "real",
    ) -> None:
        if mode not in ("real", "cache_only"):
            raise ValueError(f"Unsupported mode for OpenAI client: {mode}")
        self.mode = mode
        self.cost = cost_tracker or CostTracker()
        self.cache = diskcache.Cache(cache_path) if (cache_enabled and cache_path) else None

        # Lazy import so tests without openai work.
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Check .env and python-dotenv load_dotenv()."
            )
        self._client = OpenAI(api_key=api_key)

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        response_format_json: bool = True,
    ) -> dict[str, Any]:
        key = _cache_key(model, messages, temperature, seed)
        if self.cache is not None and key in self.cache:
            cached = self.cache[key]
            self.cost.record(0, 0, provider="openai", cache_hit=True)
            return {**cached, "cache_hit": True}

        if self.mode == "cache_only":
            raise RuntimeError(f"cache_only mode: miss for key {key[:12]}...")

        content, tokens_in, tokens_out = self._call_api(
            messages, model, temperature, max_tokens, seed, response_format_json
        )
        self.cost.record(tokens_in, tokens_out, provider="openai", cache_hit=False)
        result = {
            "content": _strip_fences(content),  # OpenAI-compatible routers may wrap JSON in ``` fences
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cache_hit": False,
            "model": model,
        }
        if self.cache is not None:
            self.cache[key] = {k: v for k, v in result.items() if k != "cache_hit"}
        return result

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=2, max=60))
    def _call_api(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        response_format_json: bool,
    ) -> tuple[str, int, int]:
        # gpt-5* are reasoning models: temperature is fixed at 1.0 by API,
        # and `max_tokens` was renamed to `max_completion_tokens`. We also
        # force `reasoning_effort='minimal'` so the model spends near-zero
        # tokens on hidden reasoning — we want the structured JSON output,
        # and our own self-consistency / DS combine handles the uncertainty.
        is_reasoning = model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3") or model.startswith("o4")
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if is_reasoning:
            kwargs["max_completion_tokens"] = max(max_tokens, 1024)
            kwargs["reasoning_effort"] = "minimal"
        else:
            kwargs["temperature"] = temperature
            # models behind an OpenAI-compatible router may think before answering
            kwargs["max_tokens"] = max(max_tokens, 8192) if os.environ.get("OPENAI_BASE_URL") else max_tokens
        if seed is not None:
            kwargs["seed"] = seed
        if response_format_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""
        usage = resp.usage
        return content, usage.prompt_tokens, usage.completion_tokens


class GeminiLLMClient:
    """Google Gemini wrapper with retry + cache. API key from GEMINI_API_KEY env.

    Mirrors OpenAILLMClient.chat() signature so it is drop-in compatible
    everywhere build_client(provider='gemini') is requested.
    """

    # Gemini 2.5 Flash pricing (USD per 1M tokens, Apr 2026):
    PRICE_IN_PER_1M = 0.075
    PRICE_OUT_PER_1M = 0.30

    def __init__(
        self,
        cache_path: str | None = ".cache/llm_gemini.sqlite",
        cache_enabled: bool = True,
        cost_tracker: CostTracker | None = None,
        mode: str = "real",
    ) -> None:
        if mode not in ("real", "cache_only"):
            raise ValueError(f"Unsupported mode for Gemini client: {mode}")
        self.mode = mode
        self.cost = cost_tracker or CostTracker()
        self.cache = diskcache.Cache(cache_path) if (cache_enabled and cache_path) else None
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Check .env and python-dotenv load_dotenv()."
            )
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self._genai = genai

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        response_format_json: bool = True,
    ) -> dict[str, Any]:
        key = _cache_key(model, messages, temperature, seed)
        if self.cache is not None and key in self.cache:
            cached = self.cache[key]
            self.cost.record(0, 0, provider="gemini", cache_hit=True)
            return {**cached, "cache_hit": True}
        if self.mode == "cache_only":
            raise RuntimeError(f"cache_only mode: miss for key {key[:12]}...")
        content, tokens_in, tokens_out = self._call_api(
            messages, model, temperature, max_tokens, response_format_json
        )
        # Manual cost recording with Gemini-specific pricing.
        usd = (tokens_in * self.PRICE_IN_PER_1M + tokens_out * self.PRICE_OUT_PER_1M) / 1_000_000
        try:
            self.cost.record(tokens_in, tokens_out, provider="gemini", cache_hit=False)
        except TypeError:
            self.cost.record(tokens_in, tokens_out)
        result = {
            "content": content,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "usd": usd,
            "cache_hit": False,
            "model": model,
        }
        if self.cache is not None:
            self.cache[key] = {k: v for k, v in result.items() if k != "cache_hit"}
        return result

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=2, max=60))
    def _call_api(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        response_format_json: bool,
    ) -> tuple[str, int, int]:
        # Flatten chat messages to Gemini's prompt+history format.
        # We treat the first 'system' message as system_instruction and the
        # rest as a single concatenated user prompt for our deterministic path.
        sys_msg = next((m["content"] for m in messages if m.get("role") == "system"), None)
        user_blocks = [m["content"] for m in messages if m.get("role") != "system"]
        prompt = "\n\n".join(user_blocks) if user_blocks else ""
        # Gemini 2.5 reasoning models eat the output-token budget on internal
        # thinking; raise the cap aggressively.  The actual response payload is
        # still tiny (a few JSON tokens), so cost is negligible.
        gen_cfg: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max(max_tokens, 1024),
        }
        if response_format_json:
            gen_cfg["response_mime_type"] = "application/json"
        gm = self._genai.GenerativeModel(
            model_name=model,
            system_instruction=sys_msg if sys_msg else None,
            generation_config=gen_cfg,
        )
        resp = gm.generate_content(prompt)
        # Be defensive: response_text may be empty when MIME validation fails.
        try:
            content = resp.text or ""
        except Exception:
            # Fallback: extract first candidate part if .text raises.
            try:
                content = resp.candidates[0].content.parts[0].text or ""
            except Exception:
                content = ""
        # Token usage
        try:
            usage = resp.usage_metadata
            tokens_in = int(getattr(usage, "prompt_token_count", 0))
            tokens_out = int(getattr(usage, "candidates_token_count", 0))
        except Exception:
            # Rough estimate if API didn't expose usage_metadata.
            tokens_in = sum(len(s.split()) for s in user_blocks) * 2
            tokens_out = len(content.split()) * 2
        return content, tokens_in, tokens_out


def _strip_fences(text: str) -> str:
    """Remove a ```json ... ``` wrapper if the model added one."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()


class _CachedClient:
    """Shared cache + bookkeeping for the second-family judges (SOICT 2026)."""

    provider = "base"

    def __init__(self, cache_path: str, cache_enabled: bool, cost_tracker: CostTracker | None, mode: str) -> None:
        if mode not in ("real", "cache_only"):
            raise ValueError(f"Unsupported mode for {self.provider} client: {mode}")
        self.mode = mode
        self.cost = cost_tracker or CostTracker()
        self.cache = diskcache.Cache(cache_path) if (cache_enabled and cache_path) else None

    def chat(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        response_format_json: bool = True,
    ) -> dict[str, Any]:
        key = _cache_key(model, messages, temperature, seed)
        if self.cache is not None and key in self.cache:
            self.cost.record(0, 0, provider=self.provider, cache_hit=True)
            return {**self.cache[key], "cache_hit": True}
        if self.mode == "cache_only":
            raise RuntimeError(f"cache_only mode: miss for key {key[:12]}...")
        content, tokens_in, tokens_out = self._call_api(messages, model, max_tokens, response_format_json)
        self.cost.record(tokens_in, tokens_out, provider=self.provider, cache_hit=False)
        result = {"content": _strip_fences(content), "tokens_in": tokens_in,
                  "tokens_out": tokens_out, "cache_hit": False, "model": model}
        if self.cache is not None:
            self.cache[key] = {k: v for k, v in result.items() if k != "cache_hit"}
        return result


class AnthropicLLMClient(_CachedClient):
    """Claude via the Anthropic SDK. Thinking is disabled (the closest setting to
    gpt-5-nano with minimal reasoning); sampling parameters are not sent because
    current Claude models reject non-default values."""

    provider = "anthropic"

    def __init__(self, cache_path: str = ".cache/llm_anthropic.sqlite", cache_enabled: bool = True,
                 cost_tracker: CostTracker | None = None, mode: str = "real") -> None:
        super().__init__(cache_path, cache_enabled, cost_tracker, mode)
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(max_retries=8)

    @retry(stop=stop_after_attempt(6), wait=wait_exponential(multiplier=1, min=2, max=60))
    def _call_api(self, messages: list[dict], model: str, max_tokens: int,
                  response_format_json: bool) -> tuple[str, int, int]:
        resp = self._client.messages.create(
            model=model,
            max_tokens=max(max_tokens, 1024),
            thinking={"type": "disabled"},
            output_config={"effort": "low"},
            system="Respond with a single JSON object only.",
            messages=[{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"],
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError("refusal")
        content = "".join(b.text for b in resp.content if b.type == "text")
        return content, resp.usage.input_tokens, resp.usage.output_tokens


class GoogleGenAIClient(_CachedClient):
    """Gemini via the google-genai SDK with minimal thinking. The temperature is
    left at the model default, as Google recommends for Gemini 3 models."""

    provider = "genai"

    def __init__(self, cache_path: str = ".cache/llm_genai.sqlite", cache_enabled: bool = True,
                 cost_tracker: CostTracker | None = None, mode: str = "real") -> None:
        super().__init__(cache_path, cache_enabled, cost_tracker, mode)
        from google import genai
        from google.genai import types

        self._types = types
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    @retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=1, min=2, max=60))
    def _call_api(self, messages: list[dict], model: str, max_tokens: int,
                  response_format_json: bool) -> tuple[str, int, int]:
        t = self._types
        contents = [t.Content(role="model" if m["role"] == "assistant" else "user",
                              parts=[t.Part(text=m["content"])]) for m in messages]
        cfg = t.GenerateContentConfig(
            max_output_tokens=max(max_tokens, 1024),
            response_mime_type="application/json" if response_format_json else None,
            thinking_config=t.ThinkingConfig(thinking_level=t.ThinkingLevel.MINIMAL),
        )
        resp = self._client.models.generate_content(model=model, contents=contents, config=cfg)
        u = resp.usage_metadata
        tokens_out = int(u.candidates_token_count or 0) + int(getattr(u, "thoughts_token_count", 0) or 0)
        return resp.text or "", int(u.prompt_token_count or 0), tokens_out


def build_client(
    mode: str | None = None,
    cache_path: str = ".cache/llm.sqlite",
    cost_tracker: CostTracker | None = None,
    provider: str | None = None,
) -> LLMClient:
    """Factory: read MA_LED_LLM_MODE env (default 'real') → return proper client.

    `provider` selects the backend: 'openai' (default), 'gemini' (legacy SDK),
    'genai' (google-genai SDK) or 'anthropic'.
    Reads `LLM_PROVIDER` env var when not given.
    """
    mode = mode or os.environ.get("MA_LED_LLM_MODE", "real")
    provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower()
    if mode == "mock":
        return MockLLMClient()
    if mode not in ("real", "cache_only"):
        raise ValueError(f"Unknown MA_LED_LLM_MODE: {mode!r}")
    if provider == "gemini":
        return GeminiLLMClient(
            cache_path=cache_path.replace("llm.sqlite", "llm_gemini.sqlite"),
            cost_tracker=cost_tracker, mode=mode,
        )
    if provider == "genai":
        return GoogleGenAIClient(cache_path=cache_path.replace("llm.sqlite", "llm_genai.sqlite"),
                                 cost_tracker=cost_tracker, mode=mode)
    if provider == "anthropic":
        return AnthropicLLMClient(cache_path=cache_path.replace("llm.sqlite", "llm_anthropic.sqlite"),
                                  cost_tracker=cost_tracker, mode=mode)
    if provider == "openai":
        return OpenAILLMClient(
            cache_path=cache_path, cost_tracker=cost_tracker, mode=mode
        )
    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")
