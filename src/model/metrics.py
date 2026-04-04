import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    per_class_f1 = f1_score(labels, predictions, average=None, labels=[0, 1, 2], zero_division=0)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
        "f1_human": float(per_class_f1[0]),
        "f1_ai_written": float(per_class_f1[1]),
        "f1_ai_paraphrased": float(per_class_f1[2]),
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
