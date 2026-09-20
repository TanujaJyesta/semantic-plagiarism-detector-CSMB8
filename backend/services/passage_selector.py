"""Deterministic selection of useful student passages for source discovery."""

from __future__ import annotations

import re
from collections.abc import Iterable


REFERENCE_PREFIXES = ("references", "bibliography", "works cited", "http://", "https://")
STOP_HEADINGS = {"abstract", "introduction", "conclusion", "acknowledgements"}


def _canonical(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def is_meaningful_passage(text: str, minimum_characters: int = 40) -> bool:
    """Reject headings, boilerplate, very short lines, and duplicate-like text."""
    clean = " ".join(text.split()).strip()
    canonical = _canonical(clean)
    words = canonical.split()
    if len(clean) < minimum_characters or len(words) < 6:
        return False
    if canonical in STOP_HEADINGS or clean.lower().startswith(REFERENCE_PREFIXES):
        return False
    if re.fullmatch(r"[\d\W_]+", clean):
        return False
    alpha_words = [word for word in words if any(char.isalpha() for char in word)]
    return len(alpha_words) >= 5


def select_passages(
    sentences: Iterable[dict],
    maximum: int = 10,
    minimum_characters: int = 30,
) -> list[dict]:
    """Return ordered, unique, structured passages suitable for focused search."""
    selected: list[dict] = []
    seen: set[str] = set()
    for sentence in sentences:
        text = " ".join(str(sentence.get("text", "")).split())
        canonical = _canonical(text)
        if not canonical or canonical in seen:
            continue
        if not is_meaningful_passage(text, minimum_characters):
            continue
        seen.add(canonical)
        selected.append(
            {
                "passage_id": len(selected) + 1,
                "position": sentence.get("sentence_id", len(selected) + 1),
                "page": sentence.get("page_number", sentence.get("page", 1)),
                "text": text,
            }
        )
        if len(selected) >= maximum:
            break

    # Relaxed fallback for concise or single-paragraph documents
    if not selected:
        for sentence in sentences:
            text = " ".join(str(sentence.get("text", "")).split())
            canonical = _canonical(text)
            if not canonical or canonical in seen or canonical in STOP_HEADINGS:
                continue
            if len(text) >= 15 and len(canonical.split()) >= 3:
                seen.add(canonical)
                selected.append({
                    "passage_id": len(selected) + 1,
                    "position": sentence.get("sentence_id", len(selected) + 1),
                    "page": sentence.get("page_number", sentence.get("page", 1)),
                    "text": text,
                })
                if len(selected) >= maximum:
                    break

    return selected
