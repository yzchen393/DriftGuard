

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .action_extraction import action_changed, extract_action

MDS_HEADS = ("action", "attack", "violation", "task")


@dataclass(frozen=True)
class MDSLabel:
    values: dict[str, int | None]
    masks: dict[str, bool]

    def as_dict(self) -> dict[str, Any]:
        return {"labels": self.values, "masks": self.masks}


def _removed(original: Any, counterfactual: Any) -> int | None:
    if original is None or counterfactual is None:
        return None
    return int(bool(original) and not bool(counterfactual))


def build_mds_label(original: Mapping[str, Any], counterfactual: Mapping[str, Any]) -> MDSLabel:
    original_action = extract_action(original.get("action", original))
    cf_action = extract_action(counterfactual.get("action", counterfactual))
    values: dict[str, int | None] = {
        "action": int(action_changed(original_action, cf_action)),
        "attack": _removed(original.get("attack"), counterfactual.get("attack")),
        "violation": _removed(original.get("violation"), counterfactual.get("violation")),
        "task": int(not bool(original.get("task_success")) and bool(counterfactual.get("task_success"))) if "task_success" in original and "task_success" in counterfactual else None,
    }
    return MDSLabel(values, {head: value is not None for head, value in values.items()})
