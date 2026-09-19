"""Environment-backed settings for the local plagiarism research prototype."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_local_env() -> None:
    """Load simple KEY=VALUE pairs from .env without overriding real env values."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.is_file():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _integer(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


def _number(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.getenv(name, default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    tavily_api_key: str | None
    firecrawl_api_key: str | None
    database_url: str
    max_file_size_mb: int
    max_search_passages: int
    max_search_queries: int
    max_search_results: int
    max_source_pages: int
    max_source_text_chars: int
    max_source_passages_per_page: int
    top_k_source_matches: int
    semantic_weight: float
    lexical_weight: float
    tfidf_direct_threshold: float
    sbert_semantic_threshold: float
    tfidf_lexical_threshold: float
    combined_high_threshold: float
    combined_likely_threshold: float
    combined_possible_threshold: float
    request_timeout: int
    tavily_search_depth: str

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def normalized_weights(self) -> tuple[float, float]:
        total = self.semantic_weight + self.lexical_weight
        if total <= 0:
            return 0.60, 0.40
        return self.semantic_weight / total, self.lexical_weight / total

    @classmethod
    def from_environment(cls) -> "Settings":
        _load_local_env()
        depth = os.getenv("TAVILY_SEARCH_DEPTH", "basic").lower()
        if depth not in {"basic", "advanced", "fast", "ultra-fast"}:
            depth = "basic"
        return cls(
            tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
            firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY") or None,
            database_url=os.getenv("DATABASE_URL", "sqlite:///./plagiarism.db"),
            max_file_size_mb=_integer("MAX_FILE_SIZE_MB", 10),
            max_search_passages=_integer("MAX_SEARCH_PASSAGES", 10),
            max_search_queries=_integer("MAX_SEARCH_QUERIES", 20),
            max_search_results=min(20, _integer("MAX_SEARCH_RESULTS", 5)),
            max_source_pages=_integer("MAX_SOURCE_PAGES", 10),
            max_source_text_chars=_integer("MAX_SOURCE_TEXT_CHARS", 50000),
            max_source_passages_per_page=_integer("MAX_SOURCE_PASSAGES_PER_PAGE", 200),
            top_k_source_matches=_integer("TOP_K_SOURCE_MATCHES", 3),
            semantic_weight=_number("SEMANTIC_WEIGHT", 0.60),
            lexical_weight=_number("LEXICAL_WEIGHT", 0.40),
            tfidf_direct_threshold=_number("TFIDF_DIRECT_THRESHOLD", 0.85),
            sbert_semantic_threshold=_number("SBERT_SEMANTIC_THRESHOLD", 0.75),
            tfidf_lexical_threshold=_number("TFIDF_LEXICAL_THRESHOLD", 0.55),
            combined_high_threshold=_number("COMBINED_HIGH_THRESHOLD", 0.85),
            combined_likely_threshold=_number("COMBINED_LIKELY_THRESHOLD", 0.70),
            combined_possible_threshold=_number("COMBINED_POSSIBLE_THRESHOLD", 0.50),
            request_timeout=_integer("REQUEST_TIMEOUT", 30),
            tavily_search_depth=depth,
        )
