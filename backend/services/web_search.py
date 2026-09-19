"""Tavily-only source discovery and provider-response normalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests

from backend.config import Settings
from backend.services.exceptions import ConfigurationError, ExternalServiceError


LOGGER = logging.getLogger(__name__)
TAVILY_SEARCH_URL = "https://api.tavily.com/search"


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float
    source_query: str


@dataclass
class SourceCandidate:
    url: str
    title: str
    search_score: float = 0.0
    matched_queries: list[str] = field(default_factory=list)
    matched_passage_ids: list[int] = field(default_factory=list)


def normalize_url(url: str) -> str | None:
    """Accept only HTTP(S) URLs and remove fragments/tracking parameters."""
    try:
        parts = urlsplit(url.strip())
    except (AttributeError, ValueError):
        return None
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return None
    query = urlencode(
        [(key, value) for key, value in parse_qsl(parts.query, keep_blank_values=True)
         if not key.lower().startswith("utm_")]
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def _error_message(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return f"Tavily returned HTTP {response.status_code}."
    detail = body.get("detail", body) if isinstance(body, dict) else body
    return f"Tavily returned HTTP {response.status_code}: {detail}"


def search_tavily(query: str, settings: Settings) -> list[SearchResult]:
    """Search Tavily and return only this application's provider-neutral fields."""
    if not settings.tavily_api_key:
        raise ConfigurationError("TAVILY_API_KEY is not configured.")
    payload = {
        "query": query,
        "search_depth": settings.tavily_search_depth,
        "max_results": settings.max_search_results,
        "chunks_per_source": 1,
        "topic": "general",
        "include_answer": False,
        "include_raw_content": False,
    }
    try:
        response = requests.post(
            TAVILY_SEARCH_URL,
            headers={"Authorization": f"Bearer {settings.tavily_api_key}"},
            json=payload,
            timeout=settings.request_timeout,
        )
    except requests.RequestException as exc:
        raise ExternalServiceError("Tavily", "Tavily search could not be reached.") from exc
    if not response.ok:
        LOGGER.error("Tavily search failed with HTTP %s", response.status_code)
        raise ExternalServiceError("Tavily", _error_message(response), response.status_code)
    try:
        body = response.json()
    except ValueError as exc:
        raise ExternalServiceError("Tavily", "Tavily returned invalid JSON.") from exc
    raw_results = body.get("results", []) if isinstance(body, dict) else []
    if not isinstance(raw_results, list):
        raise ExternalServiceError("Tavily", "Tavily returned an invalid results payload.")
    normalized: list[SearchResult] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        url = normalize_url(str(item.get("url", "")))
        if not url:
            continue
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        normalized.append(
            SearchResult(
                title=str(item.get("title", "Untitled source")).strip() or "Untitled source",
                url=url,
                snippet=str(item.get("content", "")).strip(),
                score=score,
                source_query=query,
            )
        )
    return normalized


def discover_sources(passages: list[dict], settings: Settings) -> list[SourceCandidate]:
    """Search a bounded query set and deduplicate source URLs across passages."""
    from backend.services.query_generator import generate_queries

    candidates: dict[str, SourceCandidate] = {}
    used_queries = 0
    for passage in passages:
        for query in generate_queries(passage):
            if used_queries >= settings.max_search_queries:
                break
            used_queries += 1
            for result in search_tavily(query, settings):
                candidate = candidates.get(result.url)
                if candidate is None:
                    candidate = SourceCandidate(url=result.url, title=result.title, search_score=result.score)
                    candidates[result.url] = candidate
                elif result.score > candidate.search_score:
                    candidate.search_score = result.score
                    candidate.title = result.title
                if query not in candidate.matched_queries:
                    candidate.matched_queries.append(query)
                passage_id = int(passage["passage_id"])
                if passage_id not in candidate.matched_passage_ids:
                    candidate.matched_passage_ids.append(passage_id)
        if used_queries >= settings.max_search_queries:
            break
    LOGGER.info("Tavily discovery produced %d unique source URLs", len(candidates))
    return list(candidates.values())
