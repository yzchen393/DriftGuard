from driftguard.causal.action_extraction import ActionIR, extract_action
from driftguard.causal.counterfactual import RolloutState, build_counterfactual_state
from driftguard.causal.mds import build_mds_label


def test_action_extraction_is_structural():
    assert extract_action({"tool": "lookup", "args": {"sort": "date"}}) == ActionIR(tool_or_operation="lookup", arguments={"sort": "date"})


def test_counterfactual_replaces_one_memory_and_preserves_state():
    state = RolloutState("q", memories=("a", "b"), agent_checkpoint="ckpt", seed=7)
    cf = build_counterfactual_state(state, 1, "neutral b")
    assert cf.memories == ("a", "neutral b") and cf.agent_checkpoint == state.agent_checkpoint and cf.seed == state.seed


def test_mds_masks_missing_outcome():
    label = build_mds_label({"action": {"tool": "a"}, "attack": True, "task_success": False}, {"action": {"tool": "b"}, "attack": False, "task_success": True})
    assert label.values == {"action": 1, "attack": 1, "violation": None, "task": 1}
    assert label.masks["violation"] is False
