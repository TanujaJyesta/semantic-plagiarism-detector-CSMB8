"""Train the fusion classifier on labeled (passage, source, label) pairs.

Expected input CSV columns: passage, source_text, label (1 = plagiarized, 0 = original)
Build this CSV from the PAN corpus + your controlled paraphrase set.

Usage:
    python -m backend.evaluation.train_classifier data/labeled_pairs.csv
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from backend.similarity.lexical import tfidf_similarity, word_overlap_ratio
from backend.similarity.semantic import sbert_similarity
from backend.evaluation.metrics import evaluate


def extract_features_row(passage: str, source_text: str) -> dict:
    lexical = tfidf_similarity(passage, source_text)
    semantic = sbert_similarity(passage, source_text)
    overlap = word_overlap_ratio(passage, source_text)
    length_ratio = min(len(passage), len(source_text)) / max(len(passage), len(source_text), 1)
    return {
        "lexical_score": lexical,
        "semantic_score": semantic,
        "word_overlap_ratio": overlap,
        "length_ratio": length_ratio,
    }


def main(csv_path: str):
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} labeled pairs. Extracting features (this calls SBERT for each row)...")

    feature_rows = [extract_features_row(row.passage, row.source_text) for row in df.itertuples()]
    X = pd.DataFrame(feature_rows)
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    param_grid = {"clf__C": [0.01, 0.1, 1, 10]}
    grid = GridSearchCV(pipe, param_grid, cv=5, scoring="f1")
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print("Best hyperparameters:", grid.best_params_)

    y_pred = best_model.predict(X_test)
    print("Test set performance:", evaluate(list(y_test), list(y_pred)))

    coefs = best_model.named_steps["clf"].coef_[0]
    print("\nLearned feature weights (higher = more influence on the plagiarism decision):")
    for name, coef in zip(X.columns, coefs):
        print(f"  {name}: {coef:.3f}")

    out_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fusion_classifier.joblib")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(best_model, out_path)
    print(f"\nSaved trained classifier to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m backend.evaluation.train_classifier data/labeled_pairs.csv")
        sys.exit(1)
    main(sys.argv[1])
