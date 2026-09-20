import re


def normalize_for_tfidf(text):
    """
    Normalize text for TF-IDF processing.

    This version is used only for lexical similarity.
    The original sentence is always preserved separately.
    """

    # Convert to lowercase
    text = text.lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def prepare_sentences(sentences):
    """
    Prepare extracted sentences for the plagiarism pipeline.

    Adds a normalized TF-IDF representation while preserving
    the original sentence.
    """

    prepared_sentences = []

    for sentence in sentences:

        original_text = sentence["text"]

        tfidf_text = normalize_for_tfidf(original_text)

        prepared_sentences.append({
            "sentence_id": sentence["sentence_id"],
            "page_number": sentence["page_number"],
            "text": original_text,
            "tfidf_text": tfidf_text
        })

    return prepared_sentences


def prepare_text_passages(passages):
    """Add the lexical representation to arbitrary structured passage records."""
    prepared = []
    for passage in passages:
        item = passage.copy()
        item["tfidf_text"] = normalize_for_tfidf(str(item.get("text", "")))
        prepared.append(item)
    return prepared
