from backend.services.document_processor import process_document
from backend.services.preprocessing import prepare_sentences


file_path = "data/test.pdf"


# Step 2: Extract document
sentences = process_document(file_path)


# Step 3: Prepare sentences
prepared_sentences = prepare_sentences(sentences)


print("\n===== PREPROCESSING RESULT =====\n")


for sentence in prepared_sentences:

    print("Sentence ID:", sentence["sentence_id"])
    print("Page:", sentence["page_number"])

    print("Original:")
    print(sentence["text"])

    print("\nTF-IDF version:")
    print(sentence["tfidf_text"])

    print("-" * 70)