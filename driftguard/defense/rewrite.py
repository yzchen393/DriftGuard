

from __future__ import annotations

from pathlib import Path
from typing import Callable


def _prompt_path(name: str) -> Path:
    return Path(__file__).resolve().parents[2] / "prompts" / name


def build_memory_rewrite_prompt(candidate_memory: str) -> str:
    template = _prompt_path("causal/memory_neutralization.txt").read_text(encoding="utf-8")
    return template.replace("{candidate_memory}", candidate_memory)


def build_zero_shot_risk_prompt(*, victim_agent: str, memory_intervention: str, query: str, context_without_candidate: str, candidate_memory: str, task_contract: str) -> str:
    template = _prompt_path("baselines/risk_zero_shot.txt").read_text(encoding="utf-8")
    values = {
        "{victim_agent}": victim_agent,
        "{memory_intervention}": memory_intervention,
        "{query}": query,
        "{context_without_candidate}": context_without_candidate,
        "{candidate_memory}": candidate_memory,
        "{task_contract}": task_contract,
    }
    for marker, value in values.items():
        template = template.replace(marker, value)
    return template


def apply_rewrite(memory: str, operator: Callable[[str], str]) -> str:
    prompt = build_memory_rewrite_prompt(memory)
    repaired = operator(prompt)
    if not isinstance(repaired, str) or not repaired.strip():
        raise ValueError("rewrite operator returned empty text")
    return repaired.strip()
