"""Combine lexical and semantic scores into one decision.

Two modes are supported:
1. Fixed-weight formula (fast, no training needed - use this first).
2. Trained classifier (logistic regression over multiple features -
   use this once you have a labeled dataset; see evaluation/train_classifier.py).
"""
import os
import joblib
import numpy as np

_CLASSIFIER_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "fusion_classifier.joblib")
_classifier = None


def fused_score(lexical: float, semantic: float, alpha: float = 0.6) -> float:
    """Simple weighted average. alpha favors semantic score by default -
    tune this on your validation set once you have real results."""
    return alpha * semantic + (1 - alpha) * lexical


def is_plagiarized_simple(lexical: float, semantic: float, alpha: float = 0.6, threshold: float = 0.55) -> bool:
    return fused_score(lexical, semantic, alpha) > threshold


def _load_classifier():
    global _classifier
    if _classifier is None:
        if not os.path.exists(_CLASSIFIER_PATH):
            raise FileNotFoundError(
                f"No trained classifier found at {_CLASSIFIER_PATH}. "
                "Run evaluation/train_classifier.py first, or use is_plagiarized_simple() instead."
            )
        _classifier = joblib.load(_CLASSIFIER_PATH)
    return _classifier


def is_plagiarized_classifier(lexical: float, semantic: float, word_overlap: float, length_ratio: float) -> tuple[bool, float]:
    """Use the trained classifier instead of the fixed-weight formula.
    Returns (prediction, probability_of_plagiarism)."""
    clf = _load_classifier()
    features = np.array([[lexical, semantic, word_overlap, length_ratio]])
    prediction = bool(clf.predict(features)[0])
    probability = float(clf.predict_proba(features)[0][1])
    return prediction, probability
