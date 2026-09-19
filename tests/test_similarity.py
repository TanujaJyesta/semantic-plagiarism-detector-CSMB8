import json

from backend.services.document_processor import process_document
from backend.services.preprocessing import prepare_sentences
from backend.services.similarity import calculate_all_similarities


source_file = "data/source.pdf"
submitted_file = "data/submitted.pdf"


# Process source document
source_sentences = process_document(source_file)
source_sentences = prepare_sentences(source_sentences)


# Process submitted document
submitted_sentences = process_document(submitted_file)
submitted_sentences = prepare_sentences(submitted_sentences)


# Calculate TF-IDF and SBERT similarities
results = calculate_all_similarities(
    source_sentences,
    submitted_sentences
)


# Save results as JSON
with open(
    "results/similarity_comparison.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        results,
        file,
        indent=4,
        ensure_ascii=False
    )


# Display results
print("\n===== TF-IDF vs SBERT COMPARISON =====\n")


for result in results:

    print(
        f"Source S{result['source_sentence_id']} "
        f"<-> Submitted S{result['submitted_sentence_id']}"
    )

    print(
        f"TF-IDF : {result['tfidf_score']:.4f}"
    )

    print(
        f"SBERT  : {result['sbert_score']:.4f}"
    )

    print("-" * 60)


print("\nResults saved to:")
print("results/similarity_comparison.json")