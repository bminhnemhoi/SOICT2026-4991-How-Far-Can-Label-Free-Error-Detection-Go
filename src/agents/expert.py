"""Domain Expert agent: constraint-based / cross-row logic."""

from __future__ import annotations

from src.agents.base import BaseAgent


class ExpertAgent(BaseAgent):
    role = "expert"
    prompt_path = "prompts/expert_v1.md"
    schema_path = "prompts/schemas/expert_output.json"
