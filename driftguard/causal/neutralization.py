

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Neutralizer(Protocol):
    def __call__(self, memory: str, *, context: str = "") -> str: ...


@dataclass(frozen=True)
class NeutralizationResult:
    original: str
    neutralized: str
    changed: bool
    valid: bool = True
    reason: str | None = None


def neutralize_memory(memory: str, neutralizer: Neutralizer, *, context: str = "") -> NeutralizationResult:
    if not isinstance(memory, str) or not memory.strip():
        raise ValueError("memory must be a non-empty string")
    repaired = neutralizer(memory, context=context)
    if not isinstance(repaired, str) or not repaired.strip():
        return NeutralizationResult(memory, "", changed=False, valid=False, reason="empty neutralization")
    repaired = repaired.strip()
    return NeutralizationResult(memory, repaired, changed=repaired != memory.strip())
