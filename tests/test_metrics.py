from driftguard.evaluation.detection import evaluate_detection


def test_detection_metrics_perfect_scores():
    result = evaluate_detection([0.1, 0.9], [0, 1])
    assert result["auroc"] == 1.0 and result["auprc"] == 1.0
