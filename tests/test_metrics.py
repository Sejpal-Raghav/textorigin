import numpy as np

from src.model.metrics import compute_metrics


def test_metrics_include_calibration_measures():
    logits = np.array([[4.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 1.0, 5.0]])
    labels = np.array([0, 1, 2])
    result = compute_metrics((logits, labels))
    assert 0 <= result["brier_score"] <= 2
    assert 0 <= result["expected_calibration_error"] <= 1
