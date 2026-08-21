

from __future__ import annotations

from typing import Iterable, Mapping


def summarize_defense(rows: Iterable[Mapping[str, bool]]) -> dict[str, float]:
    records = list(rows)
    if not records:
        raise ValueError("defense rows cannot be empty")
    asr = [bool(row["attack_success"]) for row in records if "attack_success" in row]
    tsr = [bool(row["task_success"]) for row in records if "task_success" in row]
    return {"asr": sum(asr) / len(asr) if asr else 0.0, "tsr": sum(tsr) / len(tsr) if tsr else 0.0, "n": float(len(records))}
