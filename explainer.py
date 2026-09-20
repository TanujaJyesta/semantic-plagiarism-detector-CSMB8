"""Turn raw similarity scores into a plain-language explanation."""


def generate_explanation(lexical: float, semantic: float) -> str:
    if lexical > 0.7 and semantic > 0.7:
        return (
            f"High lexical ({lexical:.2f}) and semantic ({semantic:.2f}) overlap - "
            "likely verbatim or near-verbatim copying."
        )
    if semantic > 0.7 and lexical < 0.5:
        return (
            f"High semantic similarity ({semantic:.2f}) but lower lexical overlap "
            f"({lexical:.2f}) - this passage was likely paraphrased from the source."
        )
    if lexical > 0.6 and semantic < 0.5:
        return (
            f"Notable word overlap ({lexical:.2f}) but low semantic similarity "
            f"({semantic:.2f}) - may be a false positive from shared vocabulary or terminology."
        )
    return "Low similarity on both measures - likely original content."


def build_evidence_record(passage: str, source_url: str, source_excerpt: str,
                           lexical: float, semantic: float, fused: float) -> dict:
    """Assemble the full record shown to the evaluator/student for one flagged passage."""
    return {
        "passage": passage,
        "source_url": source_url,
        "source_excerpt": source_excerpt[:500],
        "lexical_score": round(lexical, 3),
        "semantic_score": round(semantic, 3),
        "fused_score": round(fused, 3),
        "explanation": generate_explanation(lexical, semantic),
    }
