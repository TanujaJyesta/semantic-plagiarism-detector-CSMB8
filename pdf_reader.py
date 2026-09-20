"""Extract raw text from PDF files."""
import fitz  # PyMuPDF


def extract_text_from_pdf(file_path: str) -> str:
    """Return all text content from a PDF file, page by page."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text
