"""Extract raw text from DOCX files."""
from docx import Document


def extract_text_from_docx(file_path: str) -> str:
    """Return all paragraph text content from a DOCX file."""
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
