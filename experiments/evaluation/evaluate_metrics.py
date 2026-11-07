
import csv
import math
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix
)

# Preliminary manually assigned meaning-equivalence labels
true_labels = [1, 1, 1, 0, 0]

model_scores = {
    "TF-IDF": [1.0000, 0.3020, 0.4787, 0.0000, 0.6212],
    "MiniLM": [1.0000, 0.8341, 0.8033, 0.0919, 0.7248],
    "MPNet": [1.0000, 0.9178, 0.7981, 0.0636, 0.6134]
}

# Demonstration threshold only; not a final research threshold
threshold = 0.75

results = []

print("EVALUATION METRICS")
print("=" * 60)
print("Demonstration threshold:", threshold)

for model, scores in model_scores.items():
    predictions = [1 if score >= threshold else 0 for score in scores]

    precision = precision_score(
        true_labels, predictions, zero_division=0
    )
    recall = recall_score(
        true_labels, predictions, zero_division=0
    )
    f1 = f1_score(
        true_labels, predictions, zero_division=0
    )
    accuracy = accuracy_score(true_labels, predictions)

    tn, fp, fn, tp = confusion_matrix(
        true_labels, predictions, labels=[0, 1]
    ).ravel()

    print(f"\nModel: {model}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")

    results.append([
        model, threshold, precision, recall, f1, accuracy,
        tp, fp, fn, tn
    ])

with open(
    "experiments/evaluation/classification_metrics.csv",
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.writer(file)
    writer.writerow([
        "Model", "Threshold", "Precision", "Recall",
        "F1", "Accuracy", "TP", "FP", "FN", "TN"
    ])
    writer.writerows(results)


# PAN PlagDet formula
def calculate_plagdet(precision, recall, granularity):
    if granularity < 1:
        raise ValueError("Granularity must be at least 1.")

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return f1 / math.log2(1 + granularity)


# Illustrative calculation only, not an experimental result
example_plagdet = calculate_plagdet(
    precision=0.80,
    recall=0.75,
    granularity=1.0
)

print("\nIllustrative PAN PlagDet calculation:")
print(f"PlagDet: {example_plagdet:.4f}")

print("\nClassification results saved successfully.")
