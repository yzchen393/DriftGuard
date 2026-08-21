

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RISK_HEADS = ("action", "attack", "violation", "task")


@dataclass(frozen=True)
class RiskModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    num_heads: int = 4
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    max_length: int = 2048
    use_lora: bool = True
    local_files_only: bool = True


def _require_torch():
    try:
        import torch
        from torch import nn
    except ImportError as exc:                    
        raise ImportError("RiskModel requires torch and transformers; install requirements.txt") from exc
    return torch, nn


class RiskModel:


    def __new__(cls, *args: Any, **kwargs: Any):
        torch, nn = _require_torch()

        class _RiskModel(nn.Module):
            heads = RISK_HEADS

            def __init__(self, config: RiskModelConfig):
                super().__init__()
                try:
                    from transformers import AutoModelForCausalLM
                except ImportError as exc:                    
                    raise ImportError("RiskModel requires transformers") from exc
                self.config = config
                self.backbone = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, trust_remote_code=False, local_files_only=config.local_files_only)
                if config.use_lora:
                    try:
                        from peft import LoraConfig, TaskType, get_peft_model
                    except ImportError as exc:                    
                        raise ImportError("LoRA training requires peft") from exc
                    self.backbone = get_peft_model(self.backbone, LoraConfig(r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type=TaskType.CAUSAL_LM, bias="none"))
                hidden_size = int(self.backbone.config.hidden_size)
                self.heads = nn.ModuleDict({name: nn.Linear(hidden_size, 1) for name in RISK_HEADS})

            def forward(self, input_ids=None, attention_mask=None, **_: Any) -> dict[str, Any]:
                backbone = self.backbone.get_base_model() if hasattr(self.backbone, "get_base_model") else self.backbone
                outputs = backbone.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                hidden = outputs.last_hidden_state
                if attention_mask is None:
                    pooled = hidden[:, -1, :]
                else:
                    last = attention_mask.long().sum(dim=1).clamp_min(1) - 1
                    pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
                logits = {name: head(pooled).squeeze(-1) for name, head in self.heads.items()}
                probabilities = {name: torch.sigmoid(value) for name, value in logits.items()}
                risk = torch.stack(tuple(probabilities[name] for name in RISK_HEADS), dim=-1).amax(dim=-1)
                return {"logits": logits, "probabilities": probabilities, "risk": risk}

        return _RiskModel(*args, **kwargs)

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs: Any):
        return cls(RiskModelConfig(model_name_or_path=model_name_or_path, **kwargs))
