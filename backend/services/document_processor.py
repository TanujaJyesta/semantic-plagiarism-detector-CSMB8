import pymupdf
from docx import Document
import re
from pathlib import Path


def extract_pdf_text(file_path):
    """
    Extract text from a PDF while preserving page numbers.
    """

    pages = []

    document = pymupdf.open(file_path)

    for page_number, page in enumerate(document, start=1):

        text = page.get_text("text").strip()

        if text:
            pages.append({
                "page_number": page_number,
                "text": text
            })

    document.close()

    return pages


def extract_docx_text(file_path):
    """
    Extract text from a DOCX file.
    """

    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    complete_text = "\n".join(paragraphs).strip()

    if not complete_text:
        return []

    return [{
        "page_number": 1,
        "text": complete_text
    }]


def extract_txt_text(file_path):
    """
    Extract text from a TXT file.
    """

    with open(file_path, "r", encoding="utf-8", errors="replace") as file:
        text = file.read().strip()

    if not text:
        return []

    return [{
        "page_number": 1,
        "text": text
    }]


def extract_text(file_path):
    """
    Detect file type and extract text.
    """

    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return extract_pdf_text(file_path)

    elif extension == ".docx":
        return extract_docx_text(file_path)

    elif extension == ".txt":
        return extract_txt_text(file_path)

    else:
        raise ValueError(
            "Unsupported file format. "
            "Please upload PDF, DOCX, or TXT."
        )


def normalize_text(text):
    """
    Basic text normalization.

    We preserve the actual wording because the original
    sentence will later be displayed as plagiarism evidence.
    """

    # Replace newlines and multiple spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text):
    """
    Split text into individual sentences.
    """

    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def process_document(file_path):
    """
    Complete document processing pipeline.

    Returns structured sentence information.
    """

    pages = extract_text(file_path)

    sentences = []

    sentence_id = 1

    for page in pages:

        cleaned_page = normalize_text(page["text"])

        page_sentences = split_sentences(cleaned_page)

        for sentence in page_sentences:

            sentences.append({
                "sentence_id": sentence_id,
                "page_number": page["page_number"],
                "text": sentence
            })

            sentence_id += 1

    return sentences