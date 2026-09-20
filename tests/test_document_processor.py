from backend.services.document_processor import process_document


file_path = "data/test.pdf"

sentences = process_document(file_path)


print("\n===== DOCUMENT PROCESSING RESULT =====\n")

for sentence in sentences:

    print("Sentence ID:", sentence["sentence_id"])
    print("Page:", sentence["page_number"])
    print("Text:", sentence["text"])
    print("-" * 60)