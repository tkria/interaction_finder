"""
SearXNG search backend for privacy-respecting meta-search.

SearXNG is an open-source meta-search engine that aggregates results from
multiple search engines while protecting user privacy. This backend queries
a SearXNG instance via its JSON API.
"""

import asyncio
from typing import Dict, Any, List, Optional

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore

from interaction_finder.logging import logfire, get_logger

logger = get_logger(__name__)

from ..models import SearchBackend, SearchQuery, SearchResult


class SearXNGBackend(SearchBackend):
    """SearXNG meta-search backend using JSON API."""

    def __init__(self, config: Dict[str, Any] = {}):
        """Initialize SearXNG backend with configuration."""
        super().__init__(config)

        if not HTTPX_AVAILABLE:
            raise RuntimeError(
                "httpx is required for SearXNG backend but not installed. "
                "Install it with: pip install httpx"
            )

        self.base_url = config.get("base_url", "http://127.0.0.1:8080")
        self.timeout = config.get("timeout", 30)
        categories = config.get("categories", ["general"])
        self.categories: List[str] = (
            categories if isinstance(categories, list) else [categories]
        )
        self.engines: Optional[List[str]] = config.get("engines")
        self.language = config.get("language", "en")
        self._session: Optional[httpx.AsyncClient] = None  # type: ignore[valid-type]

    @property
    def name(self) -> str:
        """Name identifier for this backend."""
        return "searxng"

    async def _get_session(self) -> "httpx.AsyncClient":
        """Get or create HTTP session."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(timeout=self.timeout)  # type: ignore[misc]
        return self._session

    def _build_search_params(self, query: SearchQuery) -> Dict[str, str]:
        """Build query parameters for SearXNG API."""
        params: Dict[str, str] = {
            "q": query.query,
            "format": "json",
            "language": self.language,
        }
        if self.categories:
            params["categories"] = ",".join(self.categories)
        if self.engines:
            params["engines"] = ",".join(self.engines)
        return params

    def _parse_response(
        self, response_data: Dict[str, Any], query: SearchQuery
    ) -> List[SearchResult]:
        """Parse SearXNG JSON response into SearchResult objects."""
        if "error" in response_data:
            raise RuntimeError(f"SearXNG error: {response_data['error']}")
        raw_results = response_data.get("results", [])[: query.max_results]
        # Find max score for normalization (scores vary widely, e.g. 0.05 to 4.0)
        max_score = max((item.get("score", 0) for item in raw_results), default=1) or 1
        results: List[SearchResult] = []
        for item in raw_results:
            url = item.get("url", "")
            if not url:
                continue
            snippet = item.get("content", "")
            score = item.get("score")
            results.append(
                SearchResult(
                    title=item.get("title") or "(No title)",
                    url=url,
                    snippet=snippet if snippet else None,
                    relevance=score / max_score if score is not None else None,
                )
            )
        return results

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform a search with the given query."""
        query_text = query.query
        with logfire.span(
            f"SearXNG: {query_text}",
            query=query_text,
            max_results=query.max_results,
        ):
            try:
                session = await self._get_session()
                url = f"{self.base_url.rstrip('/')}/search"
                response = await session.get(
                    url, params=self._build_search_params(query)
                )
                response.raise_for_status()
                results = self._parse_response(response.json(), query)
                logger.info(
                    f"SearXNG search returned {len(results)} results",
                    extra={"query": query_text, "result_count": len(results)},
                )
                return results
            except httpx.TimeoutException:  # type: ignore[misc]
                raise RuntimeError(f"SearXNG search timed out after {self.timeout}s")
            except httpx.HTTPStatusError as e:  # type: ignore[misc]
                raise RuntimeError(f"SearXNG HTTP error: {e.response.status_code}")
            except httpx.RequestError as e:  # type: ignore[misc]
                raise RuntimeError(f"SearXNG connection error: {e}")
            except RuntimeError:
                raise
            except Exception as e:
                raise RuntimeError(f"SearXNG search failed: {e}")

    async def _async_health_check(self) -> bool:
        """Check if SearXNG instance is reachable (async)."""
        try:
            session = await self._get_session()
            response = await session.get(
                f"{self.base_url.rstrip('/')}/search",
                params={"q": "test", "format": "json"},
                timeout=10,
            )
            return response.status_code == 200
        except Exception:
            return False

    def healthy(self) -> bool:
        """Check if SearXNG instance is reachable."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return True
            return loop.run_until_complete(self._async_health_check())
        except RuntimeError:
            return asyncio.run(self._async_health_check())

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.is_closed:
            await self._session.aclose()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
