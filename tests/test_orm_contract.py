from driftguard.orm.dataset import PreActionExample, format_preaction_input


def test_preaction_formatter_excludes_future_fields():
    text = format_preaction_input(PreActionExample("q", "c", "m", "contract", metadata={"future_action": "must never be formatted"}))
    assert "future_action" not in text and "final_answer" not in text and "Candidate memory:" in text
