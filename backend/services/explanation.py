def generate_explanation(
    tfidf_score,
    sbert_score,
    match_type
):
    """
    Generate a human-readable explanation
    based on similarity scores and match type.
    """

    tfidf_percentage = tfidf_score * 100
    sbert_percentage = sbert_score * 100

    if match_type == "direct_match":

        explanation = (
            "The submitted sentence has substantial "
            "lexical and semantic similarity with the "
            "source sentence."
        )

    elif match_type == "semantic_match":

        explanation = (
            "The submitted sentence conveys a meaning "
            "similar to the source sentence, although "
            "different words or sentence structures "
            "are used."
        )

    elif match_type == "lexical_overlap":

        explanation = (
            "The submitted sentence shares considerable "
            "word-level similarity with the source sentence, "
            "but its semantic similarity is comparatively lower."
        )

    else:

        explanation = (
            "The submitted sentence shows relatively "
            "low similarity with the source sentence."
        )

    return {
        "tfidf_percentage": round(tfidf_percentage, 2),
        "sbert_percentage": round(sbert_percentage, 2),
        "explanation": explanation
    }


def create_match_explanation(match_result):
    """
    Create a complete explainable result
    for a matched source-submitted sentence pair.
    """

    tfidf_score = match_result["tfidf_score"]
    sbert_score = match_result["sbert_score"]
    match_type = match_result["match_type"]

    explanation = generate_explanation(
        tfidf_score,
        sbert_score,
        match_type
    )

    return {
        "source_sentence_id":
            match_result["source_sentence_id"],

        "source_page":
            match_result.get("source_page"),

        "source_text":
            match_result["source_text"],

        "submitted_sentence_id":
            match_result["submitted_sentence_id"],

        "submitted_page":
            match_result.get("submitted_page"),

        "submitted_text":
            match_result["submitted_text"],

        "tfidf_score":
            round(tfidf_score, 4),

        "sbert_score":
            round(sbert_score, 4),

        "tfidf_percentage":
            explanation["tfidf_percentage"],

        "sbert_percentage":
            explanation["sbert_percentage"],

        "match_type":
            match_type,

        "explanation":
            explanation["explanation"]
    }


def create_evidence_explanation(match_result):
    """Explain a web-derived match strictly from stored similarity evidence."""
    match_type = match_result["match_type"]
    risk = match_result["risk"]
    tfidf = match_result["similarity"]["tfidf"]
    sbert = match_result["similarity"]["sbert"]
    if match_type == "direct_match":
        text = "Strong lexical and semantic correspondence was found with the retrieved source passage."
    elif match_type == "semantic_match":
        text = "Strong semantic correspondence was found while lexical overlap was lower, which may indicate paraphrased wording."
    elif match_type == "lexical_overlap":
        text = "Substantial word-level overlap was found, while semantic similarity was comparatively lower."
    else:
        text = "The retrieved source passage has limited automated similarity evidence."
    return (
        f"{text} TF-IDF similarity is {tfidf * 100:.1f}%, SBERT similarity is "
        f"{sbert * 100:.1f}%, and the automated risk assessment is {risk}. "
        "This is source evidence for review, not a definitive academic or legal judgment."
    )
