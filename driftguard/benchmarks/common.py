

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

from driftguard.utils.config import load_yaml


@dataclass(frozen=True)
class BenchmarkRecord:
    benchmark: str
    split: str
    payload: Mapping[str, Any]


def benchmark_data_path(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    dataset = config.get("dataset")
    path_value = dataset.get("path") if isinstance(dataset, Mapping) else None
    if not path_value:
        raise ValueError(f"dataset.path is required in {config_path}")
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_dir():
        raise FileNotFoundError(
            "Benchmark data not found.\n"
            f"Please set dataset.path in {config_path}."
        )
    return path


def load_jsonl_records(path: str | Path, *, benchmark: str, split: str = "unknown") -> Iterator[BenchmarkRecord]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"line {line_number} of {path} is not a JSON object")
            yield BenchmarkRecord(benchmark, split, payload)
