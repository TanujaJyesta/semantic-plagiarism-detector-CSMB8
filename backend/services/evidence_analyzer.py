"""Combine retrieval and similarity signals into review-oriented evidence records."""

from __future__ import annotations

from backend.config import Settings
from backend.services.explanation import create_evidence_explanation
from backend.services.plagiarism_detector import classify_match, classify_risk


def analyze_evidence(student: dict, source: dict, tfidf_score: float, sbert_score: float, settings: Settings) -> dict:
    """Create one transparent match record without asserting plagiarism as fact."""
    semantic_weight, lexical_weight = settings.normalized_weights
    combined = semantic_weight * sbert_score + lexical_weight * tfidf_score
    match = {
        "submitted_passage": student["text"],
        "page": student.get("page"),
        "position": student.get("position"),
        "source": {"title": source["title"], "url": source["url"]},
        "matched_passage": source["text"],
        "source_position": source.get("position"),
        "similarity": {
            "tfidf": round(float(tfidf_score), 4),
            "sbert": round(float(sbert_score), 4),
            "combined": round(float(combined), 4),
        },
        "match_type": classify_match(tfidf_score, sbert_score, settings),
        "risk": classify_risk(combined, settings),
        "search_relevance": round(float(source.get("search_score", 0.0)), 4),
    }
    match["explanation"] = create_evidence_explanation(match)
    return match
