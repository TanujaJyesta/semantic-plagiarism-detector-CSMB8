"""Web search client - finds candidate source URLs for a passage."""
import os
import requests

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_CX = os.environ.get("GOOGLE_CX", "")


def make_query(passage: str, max_words: int = 14) -> str:
    """Build a distinctive search query from a passage.
    Skips the first few words (often generic openers) and takes
    a mid-passage slice, which tends to be more distinctive."""
    words = passage.split()
    start = min(3, max(0, len(words) - max_words))
    return " ".join(words[start:start + max_words])


def search_web(query: str, num_results: int = 5) -> list[str]:
    """Return candidate source URLs for a query string."""
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        raise RuntimeError(
            "Set GOOGLE_API_KEY and GOOGLE_CX environment variables "
            "(see README for how to get a free API key)."
        )
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CX, "q": query, "num": num_results}
    resp = requests.get(url, params=params, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    return [item["link"] for item in data.get("items", [])]
