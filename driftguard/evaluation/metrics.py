

from __future__ import annotations

from typing import Iterable


def _arrays(scores: Iterable[float], labels: Iterable[int]):
    try:
        import numpy as np
    except ImportError as exc:                    
        raise ImportError("metrics require numpy and scikit-learn") from exc
    values, targets = np.asarray(list(scores), dtype=float), np.asarray(list(labels), dtype=int)
    if values.size == 0 or values.size != targets.size:
        raise ValueError("scores and labels must be non-empty and equal length")
    return values, targets


def auroc(scores: Iterable[float], labels: Iterable[int]) -> float:
    from sklearn.metrics import roc_auc_score
    values, targets = _arrays(scores, labels)
    return float(roc_auc_score(targets, values))


def auprc(scores: Iterable[float], labels: Iterable[int]) -> float:
    from sklearn.metrics import average_precision_score
    values, targets = _arrays(scores, labels)
    return float(average_precision_score(targets, values))


def recall_at_fpr(scores: Iterable[float], labels: Iterable[int], *, max_fpr: float = 0.10) -> float:
    from sklearn.metrics import roc_curve
    if not 0.0 < max_fpr <= 1.0:
        raise ValueError("max_fpr must be in (0, 1]")
    values, targets = _arrays(scores, labels)
    fpr, tpr, _ = roc_curve(targets, values)
    valid = tpr[fpr <= max_fpr]
    return float(valid.max()) if valid.size else 0.0
