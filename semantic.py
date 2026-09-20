"""SBERT based semantic similarity."""
from sentence_transformers import SentenceTransformer, util

_model = None
_MODEL_NAME = "all-mpnet-base-v2"  # swap to all-MiniLM-L6-v2 for a faster, smaller model


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def sbert_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using SBERT embeddings.
    Good at catching paraphrased / reworded copying that TF-IDF misses."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    model = _get_model()
    embeddings = model.encode([text_a, text_b], convert_to_tensor=True)
    score = util.cos_sim(embeddings[0], embeddings[1])
    return float(score)
