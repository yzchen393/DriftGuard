

from __future__ import annotations

from typing import Any, Iterable

from .dataset import PreActionExample, format_preaction_input


def score_memories(model: Any, tokenizer: Any, examples: Iterable[PreActionExample], *, device: str | None = None) -> list[dict[str, Any]]:
    texts = [format_preaction_input(example) for example in examples]
    encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    if device is not None:
        encoded = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    output = model(**encoded)
    return [{"scores": {head: float(output["probabilities"][head][index].detach().cpu()) for head in output["probabilities"]}, "risk": float(output["risk"][index].detach().cpu())} for index in range(len(texts))]
