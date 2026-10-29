"""GptCoT agent: single-agent Chain-of-Thought baseline (ablation A6).

One LLM call combines auditor + expert reasoning. No multi-agent debate,
no BCL, no DS, no CADP.
"""

from __future__ import annotations

from src.agents.base import BaseAgent


class GptCotAgent(BaseAgent):
    role = "gptcot"
    prompt_path = "prompts/gptcot_v1.md"
    schema_path = "prompts/schemas/gptcot_output.json"
