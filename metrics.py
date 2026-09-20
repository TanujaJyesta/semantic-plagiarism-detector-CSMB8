"""Compute standard classification metrics for experiment comparisons."""
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def evaluate(y_true: list[int], y_pred: list[int]) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "false_positive_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "false_negative_rate": round(fn / (fn + tp), 4) if (fn + tp) else 0.0,
    }
