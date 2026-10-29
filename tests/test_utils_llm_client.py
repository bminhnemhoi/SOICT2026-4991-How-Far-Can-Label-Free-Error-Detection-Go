"""Tests for src/utils/llm_client.py (MockLLMClient path only — no real API)."""

from __future__ import annotations

import json

from src.utils.llm_client import MockLLMClient, build_client


def test_mock_returns_canned_when_key_matches():
    client = MockLLMClient()
    out = client.chat(
        messages=[{"role": "user", "content": "hello"}],
        model="gpt-4o-mini",
        temperature=0.0,
    )
    assert "content" in out
    obj = json.loads(out["content"])
    assert "p" in obj and "c" in obj
    assert out["tokens_in"] > 0
    assert out["tokens_out"] > 0


def test_build_client_mock_mode(monkeypatch):
    monkeypatch.setenv("MA_LED_LLM_MODE", "mock")
    client = build_client()
    assert isinstance(client, MockLLMClient)


def test_build_client_unknown_mode(monkeypatch):
    monkeypatch.setenv("MA_LED_LLM_MODE", "nonsense")
    import pytest

    with pytest.raises(ValueError):
        build_client()
