"""Tests for src/agents/{auditor,expert,judge}.py using MockLLMClient with canned JSON."""

from __future__ import annotations

import hashlib
import json

import pytest

from src.agents.auditor import AuditorAgent
from src.agents.expert import ExpertAgent
from src.agents.judge import JudgeAgent
from src.utils.llm_client import MockLLMClient, _cache_key


def _canned(client: MockLLMClient, messages, model, temperature, seed, content: str) -> None:
    """Pre-seed MockLLMClient.canned for a specific call signature."""
    k = _cache_key(model, messages, temperature, seed)
    client.canned[k] = content


def test_auditor_parses_valid_response():
    client = MockLLMClient()
    agent = AuditorAgent(client=client, model="gpt-4o-mini-2024-07-18", seed=42)
    payload = {"row_id": 1, "row": {"Age": 250}}
    msgs = agent.format_messages(payload)
    _canned(
        client, msgs, agent.model, agent.temperature, agent.seed,
        json.dumps({
            "label": "error",
            "confidence": 0.92,
            "explanation": "Age=250 grossly exceeds historical max.",
            "evidence_features": {
                "stat_anomalies": ["z>11"],
                "pattern_issues": [],
                "missingness_issues": []
            }
        })
    )
    out = agent.run(payload)
    assert out.label == "err"
    assert out.p == pytest.approx(0.92)
    assert out.a == 1.0  # no self-consistency
    # c is now a lexical proxy: explanation is non-empty + references "Age" and
    # "250" from payload.row -> base 0.5 + 0.1 per hit (capped at 1.0).
    assert 0.5 < out.c <= 1.0


def test_auditor_repairs_invalid_schema():
    # Agent retries with a corrective message when first response fails schema.
    # MockLLMClient on the retried (different) prompt returns its valid default,
    # so the agent should succeed instead of raising.
    client = MockLLMClient()
    agent = AuditorAgent(client=client, seed=42)
    payload = {"row_id": 1, "row": {"Age": 30}}
    msgs = agent.format_messages(payload)
    _canned(
        client, msgs, agent.model, agent.temperature, agent.seed,
        json.dumps({"label": "maybe", "confidence": 0.5, "explanation": "??",
                    "evidence_features": {"stat_anomalies": []}})  # invalid label
    )
    out = agent.run(payload)
    assert out.raw["label"] in ("error", "clean")


def test_expert_clean_label():
    client = MockLLMClient()
    agent = ExpertAgent(client=client, seed=42)
    payload = {"row_id": 2, "row": {"ZIP": "02115", "City": "Boston"}, "neighbor_rows": [], "constraints": []}
    msgs = agent.format_messages(payload)
    _canned(
        client, msgs, agent.model, agent.temperature, agent.seed,
        json.dumps({
            "label": "clean",
            "confidence": 0.75,
            "explanation": "No FD violations detected in target row.",
            "evidence_features": {
                "fd_violations": [],
                "denial_violations": [],
                "rule_violations": [],
                "cross_row_issues": []
            }
        })
    )
    out = agent.run(payload)
    assert out.label == "clean"
    assert out.p == pytest.approx(0.75)


def test_judge_maps_decision_hint_to_label():
    client = MockLLMClient()
    agent = JudgeAgent(client=client, seed=42)
    payload = {"row_id": 3}
    msgs = agent.format_messages(payload)
    _canned(
        client, msgs, agent.model, agent.temperature, agent.seed,
        json.dumps({
            "decision_hint": "error",
            "error_type": "outlier",
            "suspicious_fields": ["Age"],
            "rationale": "Auditor flagged Age=250 as extreme outlier."
        })
    )
    out = agent.run(payload)
    assert out.label == "err"
    assert out.p == 0.5  # Judge is not a voter
    assert out.raw["error_type"] == "outlier"


def test_self_consistency_a_is_agreement_fraction(monkeypatch):
    """When 3/5 samples say 'error' and 2/5 say 'clean', a = 3/5."""
    client = MockLLMClient()
    agent = AuditorAgent(client=client, seed=42, self_consistency_m=5,
                         self_consistency_temperature=0.7)
    payload = {"row_id": 1, "row": {"Age": 100}}
    msgs = agent.format_messages(payload)

    labels = ["error", "error", "error", "clean", "clean"]
    for i, lab in enumerate(labels):
        temperature = agent.temperature if i == 0 else agent.self_consistency_temperature
        seed = agent.seed if i == 0 else agent.seed + i
        content = json.dumps({
            "label": lab, "confidence": 0.8,
            "explanation": "Edge case borderline signal.",
            "evidence_features": {"stat_anomalies": [], "pattern_issues": [], "missingness_issues": []}
        })
        _canned(client, msgs, agent.model, temperature, seed, content)

    out = agent.run(payload)
    assert out.label == "err"                      # majority
    assert out.a == pytest.approx(3 / 5)


def test_auditor_captures_suspicious_fields_when_present():
    """C6 (Phiên 6 audit pass 3): Auditor surfaces `suspicious_fields` when LLM emits it."""
    client = MockLLMClient()
    agent = AuditorAgent(client=client, seed=42)
    payload = {"row_id": 1, "row": {"Age": 250, "Gender": "M"}}
    msgs = agent.format_messages(payload)
    _canned(
        client, msgs, agent.model, agent.temperature, agent.seed,
        json.dumps({
            "label": "error",
            "confidence": 0.92,
            "explanation": "Age=250 grossly exceeds historical max.",
            "suspicious_fields": ["Age"],
            "evidence_features": {
                "stat_anomalies": ["Age z>11"],
                "pattern_issues": [],
                "missingness_issues": []
            }
        })
    )
    out = agent.run(payload)
    assert out.suspicious_fields == ("Age",)
    assert out.as_dict()["suspicious_fields"] == ["Age"]


def test_expert_suspicious_fields_defaults_to_empty_when_omitted():
    """C6 backward-compat: pre-C6 LLM outputs without `suspicious_fields` still parse,
    yielding an empty tuple (no crash, no None)."""
    client = MockLLMClient()
    agent = ExpertAgent(client=client, seed=42)
    payload = {"row_id": 2, "row": {"ZIP": "02115", "City": "Boston"}, "neighbor_rows": [], "constraints": []}
    # MockLLMClient default for "expert" omits `suspicious_fields` (legacy fixture).
    out = agent.run(payload)
    assert out.suspicious_fields == ()
    assert out.as_dict()["suspicious_fields"] == []
