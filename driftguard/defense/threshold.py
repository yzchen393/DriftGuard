

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .scoring import aggregate_risk


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    f1: float
    precision: float
    recall: float
    positive_rate: float


def _f1(predicted: list[bool], labels: list[bool]) -> tuple[float, float, float]:
    tp = sum(p and y for p, y in zip(predicted, labels))
    fp = sum(p and not y for p, y in zip(predicted, labels))
    fn = sum(not p and y for p, y in zip(predicted, labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return f1, precision, recall


def select_threshold(scores: Iterable[Mapping[str, float]], labels: Iterable[bool], *, grid_size: int = 1001) -> ThresholdResult:
    risks = [aggregate_risk(item) for item in scores]
    targets = [bool(value) for value in labels]
    if len(risks) != len(targets) or not risks:
        raise ValueError("scores and labels must be non-empty and have equal length")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    best: ThresholdResult | None = None
    for index in range(grid_size):
        threshold = index / (grid_size - 1)
        predicted = [risk >= threshold for risk in risks]
        f1, precision, recall = _f1(predicted, targets)
        candidate = ThresholdResult(threshold, f1, precision, recall, sum(predicted) / len(predicted))
        if best is None or (candidate.f1, candidate.precision, -candidate.threshold) > (best.f1, best.precision, -best.threshold):
            best = candidate
    assert best is not None
    return best
