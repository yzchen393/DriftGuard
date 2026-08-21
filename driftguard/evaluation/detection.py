

from __future__ import annotations

from typing import Iterable, Mapping

from .metrics import auprc, auroc, recall_at_fpr


def evaluate_detection(scores: Iterable[float], labels: Iterable[int]) -> dict[str, float]:
    values, targets = list(scores), list(labels)
    return {"auprc": auprc(values, targets), "auroc": auroc(values, targets), "recall_at_10pct_fpr": recall_at_fpr(values, targets)}


def or_labels(rows: Iterable[Mapping[str, int | None]]) -> list[int]:
    return [int(any(value == 1 for value in row.values() if value is not None)) for row in rows]
