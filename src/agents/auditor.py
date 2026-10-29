"""Auditor agent: row-level / statistical / pattern signals."""

from __future__ import annotations

from src.agents.base import BaseAgent


class AuditorAgent(BaseAgent):
    role = "auditor"
    prompt_path = "prompts/auditor_v1.md"
    schema_path = "prompts/schemas/auditor_output.json"
