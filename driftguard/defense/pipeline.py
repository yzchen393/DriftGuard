

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Iterable

from .intervention import rewrite_memories
from .scoring import aggregate_risk


class MemoryScorer(Protocol):
    def __call__(self, memory: str, *, context: Mapping[str, Any]) -> Mapping[str, float]: ...


@dataclass(frozen=True)
class DefenseDecision:
    scores: tuple[Mapping[str, float], ...]
    risks: tuple[float, ...]
    rewritten_indices: tuple[int, ...]
    memories_after_rewrite: tuple[str, ...]


class OnlineDefensePipeline:
    def __init__(self, scorer: MemoryScorer, threshold: float, rewrite):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self.scorer = scorer
        self.threshold = threshold
        self.rewrite = rewrite

    def decide(self, memories: Iterable[str], *, context: Mapping[str, Any] | None = None) -> DefenseDecision:
        values = tuple(memories)
        scores = tuple(self.scorer(memory, context={} if context is None else context) for memory in values)
        risks = tuple(aggregate_risk(item) for item in scores)
        flagged = tuple(index for index, risk in enumerate(risks) if risk >= self.threshold)
        intervention = rewrite_memories(values, flagged, self.rewrite)
        return DefenseDecision(scores, risks, intervention.rewritten_indices, intervention.memories)
