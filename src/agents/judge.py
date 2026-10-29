"""Judge agent: generates decision_hint + error_type for ablation A6/A7 only.

Final decision in MA-LED is computed by code (BCL + DS + CADP), NOT by the Judge.
The Judge output is used only for:
  - explanation / debate trace logging
  - LLM-direct ablation (bypass BCL+DS+CADP)
"""

from __future__ import annotations

from typing import Any

from src.agents.base import AgentOutput, BaseAgent


class JudgeAgent(BaseAgent):
    role = "judge"
    prompt_path = "prompts/judge_v1.md"
    schema_path = "prompts/schemas/judge_output.json"

    def run(self, payload: dict[str, Any]) -> AgentOutput:
        """Judge schema has 'decision_hint' + 'error_type' instead of 'label'/'confidence'.

        Map: decision_hint 'error' → internal 'err', 'clean'/'abstain' → 'clean'
        with p=0.5 as a conservative default (Judge is NOT a voter in MA-LED).
        """
        messages = self.format_messages(payload)
        primary = self._call_once(messages, temperature=self.temperature, seed=self.seed)
        parsed = primary["parsed"]
        hint = parsed.get("decision_hint", "abstain")
        internal_label = "err" if hint == "error" else "clean"
        return AgentOutput(
            label=internal_label,
            p=0.5,
            c=1.0,
            a=1.0,
            raw=parsed,
            tokens_in=primary["tokens_in"],
            tokens_out=primary["tokens_out"],
            cache_hit=primary["cache_hit"],
        )
