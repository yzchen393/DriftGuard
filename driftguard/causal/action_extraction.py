

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ActionIR:
    tool_or_operation: str | None = None
    arguments: Any = None
    required_calls: tuple[str, ...] = ()
    execution_order: tuple[str, ...] = ()
    output_target: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)


def extract_action(record: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> ActionIR:

    if not isinstance(record, Mapping):
        records = list(record)
        if not records:
            return ActionIR()
        record = next((item for item in reversed(records) if any(k in item for k in ("tool", "tool_name", "action", "operation"))), records[-1])
    nested = record.get("action")
    source = {**record, **nested} if isinstance(nested, Mapping) else record
    tool = source.get("tool_or_operation") or source.get("tool_name") or source.get("tool") or source.get("operation") or source.get("action_name")
    return ActionIR(
        tool_or_operation=None if tool is None else str(tool),
        arguments=source.get("arguments", source.get("args", source.get("parameters"))),
        required_calls=_tuple(source.get("required_calls", source.get("calls"))),
        execution_order=_tuple(source.get("execution_order", source.get("order"))),
        output_target=None if source.get("output_target", source.get("target")) is None else str(source.get("output_target", source.get("target"))),
    )


def action_changed(original: ActionIR, counterfactual: ActionIR) -> bool:
    return original != counterfactual
