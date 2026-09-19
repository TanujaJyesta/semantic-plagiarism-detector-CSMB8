import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def calculate_tfidf_similarity(source_sentences, submitted_sentences):
    """
    Calculate pairwise TF-IDF cosine similarity between
    sentences from a source document and a submitted document.

    Returns:
        similarity_matrix
    """

    source_texts = [
        sentence["tfidf_text"]
        for sentence in source_sentences
    ]

    submitted_texts = [
        sentence["tfidf_text"]
        for sentence in submitted_sentences
    ]

    if not source_texts or not submitted_texts:
        return np.zeros((len(source_texts), len(submitted_texts)))

    # Combine both documents so that they use the same vocabulary
    all_texts = source_texts + submitted_texts

    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
    except ValueError:
        # Empty, punctuation-only, or stop-word-only passages have no lexical
        # vocabulary; a zero score is the useful, mathematically neutral result.
        return np.zeros((len(source_texts), len(submitted_texts)))

    source_count = len(source_texts)

    source_matrix = tfidf_matrix[:source_count]
    submitted_matrix = tfidf_matrix[source_count:]

    similarity_matrix = cosine_similarity(
        source_matrix,
        submitted_matrix
    )

    return similarity_matrix
