

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class RolloutState:
    query: str
    history: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    memories: Sequence[str] = field(default_factory=tuple)
    agent_checkpoint: str | None = None
    environment_snapshot: str | None = None
    seed: int | None = None


class RolloutBackend(Protocol):
    def rollout(self, state: RolloutState) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class CounterfactualPair:
    original: Mapping[str, Any]
    counterfactual: Mapping[str, Any]
    candidate_memory: str
    neutralized_memory: str
    state: RolloutState
    retained: bool = True
    rejection_reason: str | None = None


def build_counterfactual_state(state: RolloutState, candidate_index: int, neutralized_memory: str) -> RolloutState:
    if not 0 <= candidate_index < len(state.memories):
        raise IndexError("candidate_index is outside state.memories")
    memories = list(state.memories)
    memories[candidate_index] = neutralized_memory
    return RolloutState(state.query, tuple(state.history), tuple(memories), state.agent_checkpoint, state.environment_snapshot, state.seed)


def paired_rollout(state: RolloutState, candidate_index: int, neutralized_memory: str, backend: RolloutBackend) -> CounterfactualPair:
    cf_state = build_counterfactual_state(state, candidate_index, neutralized_memory)
    return CounterfactualPair(
        original=backend.rollout(state),
        counterfactual=backend.rollout(cf_state),
        candidate_memory=state.memories[candidate_index],
        neutralized_memory=neutralized_memory,
        state=state,
    )
