"""Firecrawl-only webpage retrieval and normalized source records."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

from backend.config import Settings
from backend.services.exceptions import ConfigurationError
from backend.services.web_search import SourceCandidate


LOGGER = logging.getLogger(__name__)
FIRECRAWL_SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"


@dataclass
class WebSource:
    url: str
    title: str
    text: str
    retrieved_at: str
    success: bool
    error: str | None = None
    search_score: float = 0.0
    matched_queries: list[str] | None = None
    matched_passage_ids: list[int] | None = None


def _failure(candidate: SourceCandidate, message: str) -> WebSource:
    return WebSource(
        url=candidate.url,
        title=candidate.title,
        text="",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        success=False,
        error=message,
        search_score=candidate.search_score,
        matched_queries=candidate.matched_queries,
        matched_passage_ids=candidate.matched_passage_ids,
    )


def scrape_source(candidate: SourceCandidate, settings: Settings) -> WebSource:
    """Request clean Markdown from Firecrawl for one already-validated URL."""
    if not settings.firecrawl_api_key:
        raise ConfigurationError("FIRECRAWL_API_KEY is not configured.")
    try:
        response = requests.post(
            FIRECRAWL_SCRAPE_URL,
            headers={
                "Authorization": f"Bearer {settings.firecrawl_api_key}",
                "Content-Type": "application/json",
            },
            json={"url": candidate.url, "formats": ["markdown"]},
            timeout=settings.request_timeout,
        )
        if not response.ok:
            return _failure(candidate, f"Firecrawl returned HTTP {response.status_code}.")
        body = response.json()
    except requests.Timeout:
        return _failure(candidate, "Firecrawl request timed out.")
    except requests.RequestException:
        return _failure(candidate, "Firecrawl request could not be completed.")
    except ValueError:
        return _failure(candidate, "Firecrawl returned invalid JSON.")

    data = body.get("data", {}) if isinstance(body, dict) else {}
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    text = str(data.get("markdown") or data.get("text") or "").strip()
    if not text:
        return _failure(candidate, "Firecrawl returned no readable text.")
    return WebSource(
        url=str(metadata.get("sourceURL") or candidate.url),
        title=str(metadata.get("title") or candidate.title),
        text=text[: settings.max_source_text_chars],
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        success=True,
        search_score=candidate.search_score,
        matched_queries=candidate.matched_queries,
        matched_passage_ids=candidate.matched_passage_ids,
    )


def retrieve_sources(candidates: list[SourceCandidate], settings: Settings) -> list[WebSource]:
    """Retrieve each unique source once, bounded by the configured page limit."""
    cache: dict[str, WebSource] = {}
    for candidate in candidates[: settings.max_source_pages]:
        if candidate.url not in cache:
            cache[candidate.url] = scrape_source(candidate, settings)
    sources = list(cache.values())
    LOGGER.info("Firecrawl retrieved %d/%d sources", sum(source.success for source in sources), len(sources))
    return sources
