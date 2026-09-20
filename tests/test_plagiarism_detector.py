from backend.services.document_processor import process_document
from backend.services.preprocessing import prepare_sentences
from backend.services.similarity import calculate_all_similarities
from backend.services.plagiarism_detector import detect_candidate_matches


source_file = "data/source.pdf"
submitted_file = "data/submitted.pdf"


# Process source document
source_sentences = process_document(source_file)
source_sentences = prepare_sentences(source_sentences)


# Process submitted document
submitted_sentences = process_document(submitted_file)
submitted_sentences = prepare_sentences(submitted_sentences)


# Calculate similarity
comparison_results = calculate_all_similarities(
    source_sentences,
    submitted_sentences
)


# Detect candidate matches
detected_matches = detect_candidate_matches(
    comparison_results
)


print("\n===== PLAGIARISM DECISION RESULTS =====\n")


for result in detected_matches:

    print(
        f"Source S{result['source_sentence_id']} "
        f"<-> Submitted S{result['submitted_sentence_id']}"
    )

    print(
        f"TF-IDF      : {result['tfidf_score']:.4f}"
    )

    print(
        f"SBERT       : {result['sbert_score']:.4f}"
    )

    print(
        f"Match Type  : {result['match_type']}"
    )

    print("-" * 70)