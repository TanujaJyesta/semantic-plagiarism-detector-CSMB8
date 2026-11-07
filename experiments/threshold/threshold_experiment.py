
import csv
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer

DATASET = Path("threshold_dataset.csv")
RESULTS = Path("threshold_results.csv")
PREDICTIONS = Path("test_predictions.csv")

MODEL_NAMES = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "MPNet": "sentence-transformers/all-mpnet-base-v2"
}


def load_dataset():
    with DATASET.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    required = {
        "submitted_passage", "source_passage",
        "is_plagiarized", "split"
    }

    if not rows or not required.issubset(rows[0].keys()):
        raise ValueError("Dataset is empty or required columns are missing.")

    for row in rows:
        row["is_plagiarized"] = int(row["is_plagiarized"])

    for split in ["train", "validation", "test"]:
        split_rows = [r for r in rows if r["split"] == split]
        labels = {r["is_plagiarized"] for r in split_rows}

        if not split_rows or labels != {0, 1}:
            raise ValueError(f"{split} must contain both classes.")

    return rows


def get_metrics(actual, predicted):
    return {
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
        "f1": f1_score(actual, predicted, zero_division=0),
        "accuracy": accuracy_score(actual, predicted)
    }


def select_threshold(actual, scores):
    candidates = np.unique(scores)
    best_threshold = None
    best_metrics = None
    best_key = (-1, -1, -1)

    for threshold in candidates:
        predicted = (scores >= threshold).astype(int)
        metrics = get_metrics(actual, predicted)

        # Prefer higher F1, then precision, then threshold.
        key = (
            metrics["f1"],
            metrics["precision"],
            float(threshold)
        )

        if key > best_key:
            best_key = key
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics


def get_labels(rows):
    return np.array([r["is_plagiarized"] for r in rows])


def tfidf_scores(train_rows, target_rows):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50000
    )

    # Fit vocabulary and IDF using training passages only.
    train_texts = (
        [r["submitted_passage"] for r in train_rows] +
        [r["source_passage"] for r in train_rows]
    )
    vectorizer.fit(train_texts)

    submitted = vectorizer.transform(
        [r["submitted_passage"] for r in target_rows]
    )
    source = vectorizer.transform(
        [r["source_passage"] for r in target_rows]
    )

    scores = np.asarray(submitted.multiply(source).sum(axis=1)).ravel()
    return scores


def sbert_scores(model, target_rows):
    submitted_texts = [r["submitted_passage"] for r in target_rows]
    source_texts = [r["source_passage"] for r in target_rows]

    submitted_embeddings = model.encode(
        submitted_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    source_embeddings = model.encode(
        source_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Normalized embeddings allow cosine similarity through dot product.
    return np.sum(
        submitted_embeddings * source_embeddings,
        axis=1
    )


def evaluate_model(name, train_rows, validation_rows, test_rows,
                   validation_scores, test_scores):
    validation_labels = get_labels(validation_rows)
    test_labels = get_labels(test_rows)

    threshold, validation_metrics = select_threshold(
        validation_labels, validation_scores
    )

    test_predictions = (test_scores >= threshold).astype(int)
    test_metrics = get_metrics(test_labels, test_predictions)

    print(f"\n{name} RESULTS")
    print("-" * 45)
    print(f"Selected threshold: {threshold:.6f}")

    print("\nValidation metrics:")
    for key, value in validation_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

    print("\nTest metrics:")
    for key, value in test_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

    result = {
        "model": name,
        "threshold": threshold,
        "validation_precision": validation_metrics["precision"],
        "validation_recall": validation_metrics["recall"],
        "validation_f1": validation_metrics["f1"],
        "validation_accuracy": validation_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_accuracy": test_metrics["accuracy"]
    }

    predictions = []

    for row, score, prediction in zip(
        test_rows, test_scores, test_predictions
    ):
        predictions.append({
            "pair_id": row["pair_id"],
            "model": name,
            "actual_label": row["is_plagiarized"],
            "similarity_score": float(score),
            "selected_threshold": threshold,
            "predicted_label": int(prediction)
        })

    return result, predictions


def main():
    print("Loading PAN 2013 dataset...")
    rows = load_dataset()

    train_rows = [r for r in rows if r["split"] == "train"]
    validation_rows = [r for r in rows if r["split"] == "validation"]
    test_rows = [r for r in rows if r["split"] == "test"]

    print("\nDataset distribution:")
    for name, data in [
        ("Train", train_rows),
        ("Validation", validation_rows),
        ("Test", test_rows)
    ]:
        print(f"{name}: {len(data)} pairs | {dict(Counter(r['is_plagiarized'] for r in data))}")

    results = []
    all_predictions = []

    print("\nRunning TF-IDF...")
    train_scores = tfidf_scores(train_rows, train_rows)
    validation_scores = tfidf_scores(train_rows, validation_rows)
    test_scores = tfidf_scores(train_rows, test_rows)

    result, predictions = evaluate_model(
        "TF-IDF", train_rows, validation_rows, test_rows,
        validation_scores, test_scores
    )
    results.append(result)
    all_predictions.extend(predictions)

    for name, model_name in MODEL_NAMES.items():
        print(f"\nLoading {name}: {model_name}")
        model = SentenceTransformer(model_name)

        print(f"Calculating {name} validation scores...")
        validation_scores = sbert_scores(model, validation_rows)

        print(f"Calculating {name} test scores...")
        test_scores = sbert_scores(model, test_rows)

        result, predictions = evaluate_model(
            name, train_rows, validation_rows, test_rows,
            validation_scores, test_scores
        )

        results.append(result)
        all_predictions.extend(predictions)

        del model

    result_columns = list(results[0].keys())

    with RESULTS.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=result_columns)
        writer.writeheader()
        writer.writerows(results)

    prediction_columns = [
        "pair_id", "model", "actual_label",
        "similarity_score", "selected_threshold", "predicted_label"
    ]

    with PREDICTIONS.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=prediction_columns)
        writer.writeheader()
        writer.writerows(all_predictions)

    print("\nEXPERIMENT COMPLETED")
    print("=" * 45)
    print("Results saved to:", RESULTS.resolve())
    print("Test predictions saved to:", PREDICTIONS.resolve())


if __name__ == "__main__":
    main()
