from backend.config import Settings


def classify_match(tfidf_score, sbert_score, settings=None):
    """
    Classify a sentence pair based on lexical and semantic similarity.

    IMPORTANT:
    These thresholds are provisional and will be calibrated
    experimentally using evaluation data.
    """

    # Defaults preserve the previous research-baseline behavior. The new web
    # workflow passes Settings so thresholds are environment-configurable.
    if settings is None:
        TFIDF_HIGH = 0.70
        SBERT_HIGH = 0.75
        TFIDF_LEXICAL = 0.70
    else:
        TFIDF_HIGH = settings.tfidf_direct_threshold
        SBERT_HIGH = settings.sbert_semantic_threshold
        TFIDF_LEXICAL = settings.tfidf_lexical_threshold

    if tfidf_score >= TFIDF_HIGH and sbert_score >= SBERT_HIGH:

        return "direct_match"

    elif tfidf_score < TFIDF_HIGH and sbert_score >= SBERT_HIGH:

        return "semantic_match"

    elif tfidf_score >= TFIDF_LEXICAL and sbert_score < SBERT_HIGH:

        return "lexical_overlap"

    else:

        return "low_similarity"


def detect_candidate_matches(comparison_results):
    """
    Classify all source-submitted sentence pairs.
    """

    detected_matches = []

    for result in comparison_results:

        tfidf_score = result["tfidf_score"]
        sbert_score = result["sbert_score"]

        match_type = classify_match(
            tfidf_score,
            sbert_score
        )

        result_with_classification = result.copy()

        result_with_classification["match_type"] = match_type

        detected_matches.append(
            result_with_classification
        )

    return detected_matches


def classify_risk(combined_score, settings):
    """Return an evidence-oriented risk level, distinct from match type."""
    if combined_score >= settings.combined_high_threshold:
        return "highly_suspicious"
    if combined_score >= settings.combined_likely_threshold:
        return "likely_plagiarism"
    if combined_score >= settings.combined_possible_threshold:
        return "possibly_similar"
    return "no_strong_evidence"
