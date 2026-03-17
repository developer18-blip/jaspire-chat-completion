"""
Web search tool using DuckDuckGo (no API key needed).
Returns both formatted context (for LLM) and structured sources (for API response).

Smart search strategy:
  - News queries → ddgs.news() for real-time headlines
  - Other queries → ddgs.text() with current date for freshness
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime

from ddgs import DDGS

logger = logging.getLogger(__name__)

# Patterns that indicate a NEWS query (needs ddgs.news())
_NEWS_PATTERNS = [
    r"\bnews\b", r"\bheadline\b", r"\bbreaking\b",
    r"\bwar\b", r"\battack\b", r"\bcrisis\b",
    r"\belection\b", r"\bscandal\b", r"\bprotest\b",
    r"\bannounce\b", r"\bstatement\b",
    r"\blatest\b.*\b(update|event|development)\b",
]
_NEWS_RE = [re.compile(p, re.IGNORECASE) for p in _NEWS_PATTERNS]


@dataclass
class SearchResults:
    """Search results with both LLM context and structured source data."""
    context: str = ""
    sources: list[dict] = field(default_factory=list)


def _is_news_query(query: str) -> bool:
    """Check if the query is asking for news/current events."""
    for pattern in _NEWS_RE:
        if pattern.search(query):
            return True
    return False


def _search_news(ddgs: DDGS, query: str, max_results: int) -> list[dict]:
    """Use ddgs.news() for fresh headlines. Returns unified format."""
    raw = ddgs.news(query, max_results=max_results)
    results = []
    for r in raw:
        results.append({
            "title": r.get("title", "No title"),
            "url": r.get("url", ""),
            "snippet": r.get("body", ""),
            "date": r.get("date", "")[:10],
        })
    return results


def _search_text(ddgs: DDGS, query: str, max_results: int) -> list[dict]:
    """Use ddgs.text() with date context for freshness."""
    # Add current date to query for better freshness
    today = datetime.now().strftime("%B %Y")
    enhanced_query = f"{query} {today}"

    raw = ddgs.text(enhanced_query, max_results=max_results)
    results = []
    for r in raw:
        results.append({
            "title": r.get("title", "No title"),
            "url": r.get("href", ""),
            "snippet": r.get("body", ""),
        })
    return results


def run_web_search(query: str, max_results: int = 5) -> SearchResults:
    """
    Search the web using DuckDuckGo.

    Automatically picks the best search method:
    - News queries → ddgs.news() for real-time headlines
    - Other queries → ddgs.text() with date context

    Returns:
        SearchResults with formatted context and structured source list.
    """
    if not query or not query.strip():
        return SearchResults()

    try:
        ddgs = DDGS()
        is_news = _is_news_query(query)

        if is_news:
            logger.info(f"[WebSearch] NEWS query: {query!r}")
            results = _search_news(ddgs, query, max_results)
        else:
            logger.info(f"[WebSearch] TEXT query: {query!r}")
            results = _search_text(ddgs, query, max_results)

        if not results:
            logger.warning("[WebSearch] No results returned")
            return SearchResults()

        formatted = []
        sources = []

        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            url = r.get("url", "")
            snippet = r.get("snippet", "")
            date = r.get("date", "")

            line = f"[{i}] {title}\nURL: {url}"
            if date:
                line += f"\nDate: {date}"
            line += f"\n{snippet}"

            formatted.append(line)
            sources.append({"title": title, "url": url, "snippet": snippet})

        context = "\n\n".join(formatted)
        logger.info(
            f"[WebSearch] Got {len(results)} results ({len(context)} chars) "
            f"[method={'news' if is_news else 'text'}]"
        )

        return SearchResults(context=context, sources=sources)

    except Exception as exc:
        logger.error(f"[WebSearch] Failed: {exc}")
        return SearchResults()
