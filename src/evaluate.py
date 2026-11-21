"""Metrics for the project's fixed binary label order."""

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def evaluate_predictions(y_true, y_pred) -> dict:
    labels = [0, 1]
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1.mean()),
        "per_class": {
            str(label): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix_labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
