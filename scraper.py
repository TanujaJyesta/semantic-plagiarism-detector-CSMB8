"""Fetch and clean the main text content of a web page."""
import requests
from bs4 import BeautifulSoup

# Simple in-memory cache so we never scrape the same URL twice in one run
_cache: dict[str, str] = {}


def scrape_main_text(url: str, timeout: int = 8) -> str:
    if url in _cache:
        return _cache[url]
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
    except Exception:
        text = ""
    _cache[url] = text
    return text
