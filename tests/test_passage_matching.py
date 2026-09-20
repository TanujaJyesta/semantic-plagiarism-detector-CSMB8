from backend.services.document_processor import process_document
from backend.services.preprocessing import prepare_sentences
from backend.services.similarity import calculate_all_similarities
from backend.services.passage_matcher import find_best_matches
from backend.services.plagiarism_detector import classify_match


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


# Find best source sentence for every submitted sentence
best_matches = find_best_matches(
    comparison_results,
    minimum_similarity=0.0
)


print("\n===== BEST MATCH RESULTS =====\n")


for result in best_matches:

    match_type = classify_match(
        result["tfidf_score"],
        result["sbert_score"]
    )

    print(
        f"Submitted S{result['submitted_sentence_id']}"
    )

    print(
        f"Submitted Text: "
        f"{result['submitted_text']}"
    )

    print()

    print(
        f"Best Source S{result['source_sentence_id']}"
    )

    print(
        f"Source Text: "
        f"{result['source_text']}"
    )

    print()

    print(
        f"TF-IDF Score : "
        f"{result['tfidf_score']:.4f}"
    )

    print(
        f"SBERT Score  : "
        f"{result['sbert_score']:.4f}"
    )

    print(
        f"Match Type   : "
        f"{match_type}"
    )

    print("-" * 80)