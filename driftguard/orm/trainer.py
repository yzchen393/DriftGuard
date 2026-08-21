

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .model import RISK_HEADS


@dataclass(frozen=True)
class TrainConfig:
    optimizer: str = "AdamW"
    scheduler: str = "cosine"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 3
    effective_batch_size: int = 16
    early_stopping_metric: str = "validation_auprc"


def masked_bce_loss(logits: Mapping[str, Any], labels: Mapping[str, Any], masks: Mapping[str, Any] | None = None):
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:                    
        raise ImportError("masked_bce_loss requires torch") from exc
    losses = []
    for head in RISK_HEADS:
        if head not in logits or head not in labels:
            continue
        target = labels[head].float()
        mask = masks[head].bool() if masks and head in masks else torch.ones_like(target, dtype=torch.bool)
        if mask.any():
            losses.append(F.binary_cross_entropy_with_logits(logits[head][mask], target[mask]))
    if not losses:
        raise ValueError("no valid masked heads in batch")
    return torch.stack(losses).mean()
