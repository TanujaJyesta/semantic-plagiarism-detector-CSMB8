import logging

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from backend.services.exceptions import ModelUnavailableError


# The pretrained SBERT model is initialized lazily once per Python process.
MODEL_NAME = "all-MiniLM-L6-v2"
_model = None
_model_error = None
LOGGER = logging.getLogger(__name__)


def get_model():
    """Return the process-cached CPU model or a clear initialization error."""
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error is not None:
        raise ModelUnavailableError("SBERT model is unavailable.") from _model_error
    try:
        try:
            _model = SentenceTransformer(MODEL_NAME, device="cpu", local_files_only=True)
        except Exception:
            _model = SentenceTransformer(MODEL_NAME, device="cpu")
    except Exception as exc:  # Provider/runtime errors vary by installed versions.
        _model_error = exc
        LOGGER.exception("Could not initialize the SBERT model")
        raise ModelUnavailableError("SBERT model could not be initialized.") from exc
    return _model


def sbert_available() -> bool:
    """Report whether the model has not failed initialization (without loading it)."""
    return _model_error is None


def generate_embeddings(sentences):
    """
    Generate SBERT embeddings for a list of sentences.
    """

    texts = [
        sentence["text"]
        for sentence in sentences
    ]

    if not texts:
        return np.empty((0, 0))

    embeddings = get_model().encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


def calculate_sbert_similarity(source_sentences, submitted_sentences):
    """
    Calculate pairwise semantic similarity between
    source and submitted document sentences.
    """

    if not source_sentences or not submitted_sentences:
        return np.zeros((len(source_sentences), len(submitted_sentences)))

    source_embeddings = generate_embeddings(source_sentences)

    submitted_embeddings = generate_embeddings(submitted_sentences)

    similarity_matrix = cosine_similarity(
        source_embeddings,
        submitted_embeddings
    )

    return similarity_matrix
