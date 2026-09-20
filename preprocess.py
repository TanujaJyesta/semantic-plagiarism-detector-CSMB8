"""Sentence segmentation and passage windowing."""
import re
import spacy

_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def clean_text(text: str) -> str:
    """Remove excessive whitespace and common page-artifact noise."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Page \d+ of \d+", "", text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    nlp = _get_nlp()
    doc = nlp(clean_text(text))
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def build_passages(sentences: list[str], window: int = 4, stride: int = 2) -> list[str]:
    """Group sentences into overlapping windows so multi-sentence
    plagiarism (not just single sentences) can be detected."""
    passages = []
    for i in range(0, len(sentences), stride):
        chunk = sentences[i:i + window]
        if chunk:
            passages.append(" ".join(chunk))
        if i + window >= len(sentences):
            break
    return passages


def extract_passages_from_file(file_path: str, window: int = 4, stride: int = 2) -> list[str]:
    """Convenience wrapper: file path -> list of candidate passages."""
    if file_path.lower().endswith(".pdf"):
        from .pdf_reader import extract_text_from_pdf
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        from .docx_reader import extract_text_from_docx
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    sentences = split_sentences(text)
    return build_passages(sentences, window=window, stride=stride)
