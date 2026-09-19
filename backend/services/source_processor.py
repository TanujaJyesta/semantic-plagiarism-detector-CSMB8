"""Clean retrieved source text and turn it into bounded comparison passages."""

from __future__ import annotations

import re

from backend.services.document_processor import normalize_text, split_sentences
from backend.services.web_content import WebSource


def clean_source_text(text: str, maximum_characters: int) -> str:
    """Remove common Markdown/navigation noise without claiming perfect extraction."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    lines = []
    for line in text.splitlines():
        clean = line.strip().lstrip("#>- ").strip()
        if len(clean) >= 20 and not clean.lower().startswith(("cookie", "privacy", "subscribe")):
            lines.append(clean)
    return normalize_text(" ".join(lines))[:maximum_characters]


def segment_sources(sources: list[WebSource], maximum_passages_per_page: int, maximum_characters: int) -> list[dict]:
    """Return sentence-level source passages, retaining their URL/title provenance."""
    passages: list[dict] = []
    for source_index, source in enumerate(sources, start=1):
        if not source.success:
            continue
        clean = clean_source_text(source.text, maximum_characters)
        seen: set[str] = set()
        for position, sentence in enumerate(split_sentences(clean), start=1):
            sentence = sentence.strip()
            key = sentence.lower()
            if len(sentence) < 20 or key in seen:
                continue
            seen.add(key)
            passages.append(
                {
                    "source_id": source_index,
                    "position": position,
                    "text": sentence,
                    "url": source.url,
                    "title": source.title,
                    "search_score": source.search_score,
                }
            )
            if len(seen) >= maximum_passages_per_page:
                break
    return passages
