from backend.services.document_processor import process_document
from backend.services.preprocessing import prepare_sentences
from backend.services.sbert_model import calculate_sbert_similarity


source_file = "data/source.pdf"
submitted_file = "data/submitted.pdf"


# Process source document
source_sentences = process_document(source_file)
source_sentences = prepare_sentences(source_sentences)


# Process submitted document
submitted_sentences = process_document(submitted_file)
submitted_sentences = prepare_sentences(submitted_sentences)


# Calculate SBERT semantic similarity
similarity_matrix = calculate_sbert_similarity(
    source_sentences,
    submitted_sentences
)


print("\n===== SBERT SEMANTIC SIMILARITY =====\n")


for i, source_sentence in enumerate(source_sentences):

    for j, submitted_sentence in enumerate(submitted_sentences):

        score = similarity_matrix[i][j]

        print(
            f"Source S{source_sentence['sentence_id']} "
            f"<-> Submitted S{submitted_sentence['sentence_id']} "
            f"= {score:.4f}"
        )