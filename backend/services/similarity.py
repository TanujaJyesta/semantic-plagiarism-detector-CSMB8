from backend.services.tfidf_model import calculate_tfidf_similarity
from backend.services.sbert_model import calculate_sbert_similarity
from backend.services.preprocessing import prepare_text_passages


def calculate_all_similarities(
    source_sentences,
    submitted_sentences
):
    """
    Calculate both TF-IDF and SBERT similarity
    for every source-submitted sentence pair.
    """

    tfidf_matrix = calculate_tfidf_similarity(
        source_sentences,
        submitted_sentences
    )

    sbert_matrix = calculate_sbert_similarity(
        source_sentences,
        submitted_sentences
    )

    comparison_results = []

    for i, source_sentence in enumerate(source_sentences):

        for j, submitted_sentence in enumerate(submitted_sentences):

            comparison_results.append({

                "source_sentence_id":
                    source_sentence["sentence_id"],

                "source_page":
                    source_sentence["page_number"],

                "source_text":
                    source_sentence["text"],

                "submitted_sentence_id":
                    submitted_sentence["sentence_id"],

                "submitted_page":
                    submitted_sentence["page_number"],

                "submitted_text":
                    submitted_sentence["text"],

                "tfidf_score":
                    float(tfidf_matrix[i][j]),

                "sbert_score":
                    float(sbert_matrix[i][j])
            })

    return comparison_results


def calculate_top_source_matches(student_passages, source_passages, top_k, settings):
    """Batch-encode both sides once, then analyze only top semantic candidates."""
    import numpy as np

    from backend.services.evidence_analyzer import analyze_evidence

    if not student_passages or not source_passages:
        return []
    prepared_students = prepare_text_passages(student_passages)
    prepared_sources = prepare_text_passages(source_passages)
    # Both functions batch their input; this avoids one model load per pair.
    sbert_matrix = calculate_sbert_similarity(prepared_sources, prepared_students)
    tfidf_matrix = calculate_tfidf_similarity(prepared_sources, prepared_students)
    matches = []
    for student_index, student in enumerate(student_passages):
        scores = sbert_matrix[:, student_index]
        count = min(top_k, len(source_passages))
        candidate_indices = np.argsort(scores)[-count:][::-1]
        for source_index in candidate_indices:
            match = analyze_evidence(
                student,
                source_passages[int(source_index)],
                float(tfidf_matrix[int(source_index), student_index]),
                float(sbert_matrix[int(source_index), student_index]),
                settings,
            )
            matches.append(match)
    return matches
