import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
    targets = np.eye(3)[labels]
    brier_score = np.mean(np.sum((probabilities - targets) ** 2, axis=1))
    confidence = probabilities.max(axis=1)
    correct = predictions == labels
    expected_calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        in_bin = (confidence > lower) & (confidence <= upper)
        if in_bin.any():
            expected_calibration_error += in_bin.mean() * abs(
                correct[in_bin].mean() - confidence[in_bin].mean()
            )

    per_class_f1 = f1_score(labels, predictions, average=None, labels=[0, 1, 2], zero_division=0)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "f1_human": float(per_class_f1[0]),
        "f1_ai_written": float(per_class_f1[1]),
        "f1_ai_paraphrased": float(per_class_f1[2]),
        "brier_score": float(brier_score),
        "expected_calibration_error": float(expected_calibration_error),
    }


def print_confusion_matrix(labels: list, predictions: list) -> np.ndarray:
    class_names = ["human", "ai_written", "ai_paraphrased"]
    cm = confusion_matrix(labels, predictions)

    col_width = 18
    header = f"{'':20}" + "".join(f"{c:{col_width}}" for c in class_names)
    print(header)
    print("-" * (20 + col_width * 3))
    for i, row in enumerate(cm):
        print(f"{class_names[i]:20}" + "".join(f"{v:{col_width}}" for v in row))

    return cm
