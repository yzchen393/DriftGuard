

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

RISK_HEADS = ("action", "attack", "violation", "task")


@dataclass(frozen=True)
class RiskScores:
    scores: Mapping[str, float]

    @property
    def risk(self) -> float:
        return aggregate_risk(self.scores)


def aggregate_risk(scores: Mapping[str, float]) -> float:
    available = [float(scores[head]) for head in RISK_HEADS if head in scores]
    if not available:
        raise ValueError("scores must contain at least one DriftGuard head")
    if any(value < 0.0 or value > 1.0 for value in available):
        raise ValueError("head probabilities must be in [0, 1]")
    return max(available)


def score_rewrite(scores: Iterable[Mapping[str, float]], threshold: float) -> list[bool]:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    return [aggregate_risk(item) >= threshold for item in scores]
