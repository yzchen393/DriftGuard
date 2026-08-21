from driftguard.defense.pipeline import OnlineDefensePipeline
from driftguard.defense.rewrite import build_memory_rewrite_prompt
from driftguard.defense.threshold import select_threshold


def test_online_pipeline_rewrites_flagged_memories():
    def scorer(memory, *, context):
        return {"action": 0.9 if memory == "bad" else 0.1, "attack": 0.0, "violation": 0.0, "task": 0.0}
    decision = OnlineDefensePipeline(scorer, 0.5, rewrite=lambda text: "rewritten").decide(["good", "bad"])
    assert decision.rewritten_indices == (1,) and decision.memories_after_rewrite == ("good", "rewritten")


def test_threshold_uses_validation_labels():
    result = select_threshold([{"action": value, "attack": 0.0, "violation": 0.0, "task": 0.0} for value in (0.1, 0.2, 0.8, 0.9)], [False, False, True, True])
    assert result.f1 == 1.0 and 0.2 < result.threshold <= 0.8


def test_rewrite_prompt_is_appendix_prompt():
    prompt = build_memory_rewrite_prompt("candidate memory")
    assert "You are a memory-repair assistant" in prompt
    assert "Rewrite only the provided memory" in prompt
    assert "candidate memory" in prompt
