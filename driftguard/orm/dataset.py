

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

RISK_HEADS = ("action", "attack", "violation", "task")
FORBIDDEN_INPUT_FIELDS = {"planned_action", "future_action", "tool_result", "future_tool_result", "final_answer", "attack_outcome", "violation_outcome", "task_outcome", "labels", "masks", "sample_weight"}


@dataclass(frozen=True)
class PreActionExample:
    query: str
    context_without_candidate_memory: str
    candidate_memory: str
    task_contract: str = ""
    labels: Mapping[str, int | None] | None = None
    masks: Mapping[str, bool] | None = None
    metadata: Mapping[str, Any] | None = None


def _text(value: Any) -> str:
    if value is None or isinstance(value, str):
        return "" if value is None else value
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def format_preaction_input(example: PreActionExample) -> str:
    return "\n\n".join(f"{name}:\n{_text(value)}" for name, value in (("Query", example.query), ("Context without candidate memory", example.context_without_candidate_memory), ("Candidate memory", example.candidate_memory), ("Task contract", example.task_contract)))


def example_from_mapping(row: Mapping[str, Any]) -> PreActionExample:
    return PreActionExample(
        _text(row.get("query", row.get("task", ""))),
        _text(row.get("context_without_candidate_memory", row.get("context", ""))),
        _text(row.get("candidate_memory", row.get("memory", ""))),
        _text(row.get("task_contract", row.get("contract", ""))),
        row.get("labels"), row.get("masks"),
        {key: value for key, value in row.items() if key not in FORBIDDEN_INPUT_FIELDS},
    )


class PreActionDataset:
    def __init__(self, examples: Iterable[PreActionExample]):
        self.examples = list(examples)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "PreActionDataset":
        rows = []
        with Path(path).open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(example_from_mapping(json.loads(line)))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
        return cls(rows)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PreActionExample:
        return self.examples[index]
