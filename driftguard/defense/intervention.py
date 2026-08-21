

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class InterventionResult:
    memories: tuple[str, ...]
    rewritten_indices: tuple[int, ...]


def rewrite_memories(memories: Iterable[str], flagged_indices: Iterable[int], rewrite: Callable[[str], str]) -> InterventionResult:
    values = tuple(memories)
    indices = tuple(sorted(set(int(index) for index in flagged_indices)))
    if any(index < 0 or index >= len(values) for index in indices):
        raise IndexError("flagged memory index is outside the retrieved memory list")
    rewritten = list(values)
    for index in indices:
        rewritten[index] = rewrite(rewritten[index])
    return InterventionResult(tuple(rewritten), indices)

