"""TF-IDF based lexical similarity."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tfidf_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using TF-IDF vectors.
    Good at catching verbatim / near-verbatim copying."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform([text_a, text_b])
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return float(score)


def word_overlap_ratio(text_a: str, text_b: str) -> float:
    """Simple extra feature: fraction of text_a's words also in text_b."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a:
        return 0.0
    return len(words_a & words_b) / len(words_a)
