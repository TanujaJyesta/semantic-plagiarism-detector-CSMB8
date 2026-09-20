"""Small deterministic search-query generator for selected passages."""

from __future__ import annotations

import re


ACADEMIC_STOPWORDS = {
    "about", "after", "again", "also", "among", "because", "being", "between",
    "could", "does", "from", "have", "into", "more", "most", "other", "should",
    "that", "their", "there", "these", "this", "those", "through", "using", "were",
    "what", "when", "which", "with", "would", "your", "project", "paper", "study",
    "author", "authors", "abstract", "figure", "table", "section", "results",
    "introduction", "conclusion", "method", "methods", "shown", "shows", "based",
}


def _clean_text(text: str, limit: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip().strip('"')
    return clean[:limit].rsplit(" ", 1)[0] if len(clean) > limit else clean


def keyword_query(text: str, maximum_words: int = 10) -> str:
    """Create a stable keyword query without NLP models or provider-specific logic."""
    terms = re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", text.lower())
    unique: list[str] = []
    for term in terms:
        if term not in ACADEMIC_STOPWORDS and term not in unique:
            unique.append(term)
        if len(unique) == maximum_words:
            break
    return " ".join(unique)


def generate_queries(passage: dict, maximum_queries: int = 2) -> list[str]:
    """Produce search queries optimized for public web & academic source discovery.

    Strategy 1: Unquoted natural query (allows semantic / token search to find papers/articles).
    Strategy 2: Compact quoted n-gram (6-8 words) for verbatim match discovery.
    Strategy 3: Distinct keyword query for domain-specific matching.
    """
    text = str(passage.get("text", "")).strip()
    clean = re.sub(r"\s+", " ", text).strip().strip('"')
    if not clean:
        return []

    queries: list[str] = []

    # 1. Natural unquoted query (up to 140 chars)
    natural = _clean_text(clean, 140)
    if natural:
        queries.append(natural)

    # 2. Compact quoted phrase (first 7-9 salient words)
    words = clean.split()
    if len(words) >= 6 and maximum_queries > len(queries):
        phrase = " ".join(words[:min(8, len(words))])
        queries.append(f'"{phrase}"')

    # 3. High-information domain keywords
    if len(queries) < maximum_queries:
        kw = keyword_query(clean)
        if kw and kw not in queries:
            queries.append(kw)

    return queries[:maximum_queries]
