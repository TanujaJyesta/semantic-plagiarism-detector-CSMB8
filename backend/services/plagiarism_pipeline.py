import json

from backend.services.document_processor import process_document
from backend.services.preprocessing import prepare_sentences
from backend.services.similarity import calculate_all_similarities
from backend.services.passage_matcher import find_best_matches
from backend.services.plagiarism_detector import classify_match
from backend.services.explanation import create_match_explanation
from backend.config import Settings
from backend.services.passage_selector import select_passages
from backend.services.similarity import calculate_top_source_matches
from backend.services.source_processor import segment_sources
from backend.services.web_content import retrieve_sources
from backend.services.web_search import discover_sources


def analyze_documents(
    source_file,
    submitted_file,
    minimum_similarity=0.0
):
    """
    Complete plagiarism analysis pipeline.

    Flow:
    1. Extract text from documents
    2. Preprocess sentences
    3. Calculate TF-IDF and SBERT similarity
    4. Find best source match for each submitted sentence
    5. Classify matches
    6. Generate explanations
    """

    # --------------------------------------------------
    # STEP 1: Process source document
    # --------------------------------------------------

    source_sentences = process_document(source_file)

    source_sentences = prepare_sentences(
        source_sentences
    )


    # --------------------------------------------------
    # STEP 2: Process submitted document
    # --------------------------------------------------

    submitted_sentences = process_document(
        submitted_file
    )

    submitted_sentences = prepare_sentences(
        submitted_sentences
    )


    # --------------------------------------------------
    # STEP 3: Calculate similarities
    # --------------------------------------------------

    comparison_results = calculate_all_similarities(
        source_sentences,
        submitted_sentences
    )


    # --------------------------------------------------
    # STEP 4: Find best matches
    # --------------------------------------------------

    best_matches = find_best_matches(
        comparison_results,
        minimum_similarity=minimum_similarity
    )


    # --------------------------------------------------
    # STEP 5: Classify + Explain
    # --------------------------------------------------

    final_matches = []

    for match in best_matches:

        match_type = classify_match(
            match["tfidf_score"],
            match["sbert_score"]
        )

        # Add classification
        match["match_type"] = match_type

        # Generate explanation
        explanation_result = create_match_explanation(
            match
        )

        final_matches.append(
            explanation_result
        )


    # --------------------------------------------------
    # STEP 6: Create final result
    # --------------------------------------------------

    result = {

        "source_document": source_file,

        "submitted_document": submitted_file,

        "source_sentence_count":
            len(source_sentences),

        "submitted_sentence_count":
            len(submitted_sentences),

        "matched_sentence_count":
            len(final_matches),

        "matches":
            final_matches
    }


    return result


def save_analysis_result(
    result,
    output_file="results/final_analysis.json"
):
    """
    Save complete analysis result as JSON.
    """

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )


def _overall_risk(matches):
    order = {
        "no_strong_evidence": 0,
        "possibly_similar": 1,
        "likely_plagiarism": 2,
        "highly_suspicious": 3,
    }
    return max((match["risk"] for match in matches), key=lambda risk: order[risk], default="no_strong_evidence")


def analyze_web_document(document_path, filename, settings=None):
    """Run one-document public-web retrieval and evidence analysis.

    ``analyze_documents`` remains above for the original two-document research
    baseline; this function is the primary user-facing workflow.
    """
    settings = settings or Settings.from_environment()
    sentences = process_document(document_path)
    if not sentences:
        raise ValueError("The document did not contain extractable text.")
    passages = select_passages(sentences, maximum=settings.max_search_passages)
    if not passages:
        raise ValueError("The document did not contain meaningful passages for analysis.")

    candidates = discover_sources(passages, settings)
    warnings = []
    sources = []
    all_matches = []
    max_sim = 0.0
    if candidates:
        sources = retrieve_sources(candidates, settings)
        failed = [source for source in sources if not source.success]
        if failed:
            warnings.append(f"{len(failed)} source page(s) could not be retrieved.")
        source_passages = segment_sources(
            sources, settings.max_source_passages_per_page, settings.max_source_text_chars
        )
        if source_passages:
            all_matches = calculate_top_source_matches(
                passages, source_passages, settings.top_k_source_matches, settings
            )
            evidence_matches = [
                match for match in all_matches if match["risk"] != "no_strong_evidence"
            ]
            if all_matches:
                max_sim = max(m["similarity"]["combined"] for m in all_matches)
        else:
            warnings.append("No readable source passages were available for comparison.")
    else:
        warnings.append("No public source candidates were found for the selected passages.")

    return {
        "filename": filename,
        "status": "completed_with_warnings" if warnings else "completed",
        "summary": {
            "total_passages": len(sentences),
            "searched_passages": len(passages),
            "sources_found": len(candidates),
            "sources_retrieved": sum(source.success for source in sources),
            "suspicious_matches": len(evidence_matches),
            "overall_risk": _overall_risk(evidence_matches),
            "highest_similarity": round(float(max_sim), 4),
        },
        "matches": evidence_matches,
        "all_comparisons": all_matches[:20],
        "sources": [
            {
                "title": source.title,
                "url": source.url,
                "retrieval_status": "success" if source.success else "failed",
                "error": source.error,
                "search_score": source.search_score,
            }
            for source in sources
        ],
        "warnings": warnings,
    }
