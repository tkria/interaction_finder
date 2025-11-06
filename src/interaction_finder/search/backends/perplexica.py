"""
Perplexica search backend for AI-powered web search.

This module provides a simplified SearchBackend implementation for querying
Perplexica, an AI-powered search engine running locally.
"""

import asyncio
from typing import Dict, Any, List, Optional

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore

from interaction_finder.logging import logfire
from ..models import (
    SearchBackend,
    SearchQuery,
    SearchResult,
)


class PerplexicaBackend(SearchBackend):
    """Perplexica search backend using local API instance."""

    def __init__(self, config: Dict[str, Any] = {}):
        """Initialize Perplexica backend with configuration."""
        super().__init__(config)

        if not HTTPX_AVAILABLE:
            raise RuntimeError(
                "httpx is required for Perplexica backend but not installed. "
                "Install it with: pip install httpx"
            )

        self.base_url = config.get("base_url", "http://localhost:3000")
        self.timeout = config.get("timeout", 60)
        self.search_mode = config.get("search_mode", "webSearch")

        # Model configurations
        self.chat_model = config.get(
            "chat_model", {"provider": "openai", "name": "gpt-4o-mini"}
        )
        self.embedding_model = config.get(
            "embedding_model", {"provider": "openai", "name": "text-embedding-3-large"}
        )

        # Session for connection reuse
        self._session: Optional[httpx.AsyncClient] = None  # type: ignore[valid-type]

    @property
    def name(self) -> str:
        """Name identifier for this backend."""
        return "perplexica"

    async def _get_session(self) -> httpx.AsyncClient:  # type: ignore[valid-type]
        """Get or create HTTP session."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(timeout=self.timeout)  # type: ignore[misc]
        return self._session

    def _build_search_request(self, query: SearchQuery) -> Dict[str, Any]:
        """Build the request payload for Perplexica API."""
        request_data = {
            "chatModel": self.chat_model,
            "embeddingModel": self.embedding_model,
            "optimizationMode": "balanced",
            "focusMode": self.search_mode,
            "query": query.query,
            "stream": False,  # Use non-streaming for simpler processing
        }

        # Add system instructions for academic searches
        if self.search_mode == "academicSearch":
            request_data["systemInstructions"] = (
                "Focus on peer-reviewed scientific literature, research papers, "
                "and academic sources."
            )

        return request_data

    def _parse_perplexica_response(
        self, response_data: Dict[str, Any], query: SearchQuery
    ) -> List[SearchResult]:
        """Parse Perplexica API response into SearchResult objects."""
        results = []
        sources = response_data.get("sources", [])

        # Limit sources based on max_results
        limited_sources = sources[: query.max_results]

        for i, source in enumerate(limited_sources):
            try:
                result = self._convert_source_to_result(source, i)
                if result:
                    results.append(result)
            except Exception as e:
                # Log conversion error but don't fail the whole search
                print(f"Warning: Failed to convert Perplexica source to result: {e}")
                continue

        return results

    def _convert_source_to_result(
        self, source: Dict[str, Any], index: int
    ) -> Optional[SearchResult]:
        """Convert a Perplexica source to SearchResult."""
        metadata = source.get("metadata", {})
        page_content = source.get("pageContent", "")

        title = metadata.get("title", "")
        url = metadata.get("url", "")

        if not url or not title:
            return None

        # Use page content as snippet if available
        snippet = page_content if page_content else None

        # Simple relevance score based on position
        relevance = max(0.1, 1.0 - (index * 0.05))

        return SearchResult(
            title=title,
            url=url,
            snippet=snippet,
            relevance=relevance,
        )

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform a search with the given query."""
        with logfire.span(
            "PerplexicaBackend.search",
            query=query.query[:100],
            max_results=query.max_results,
        ):
            logfire.info(f"Searching Perplexica: {query.query[:100]}...")
            try:
                request_data = self._build_search_request(query)
                session = await self._get_session()
                url = f"{self.base_url}/api/search"

                response = await session.post(url, json=request_data)

                # Handle HTTP errors
                if response.status_code == 500:
                    logfire.error("Perplexica internal server error")
                    raise RuntimeError("Perplexica internal server error")
                elif response.status_code != 200:
                    logfire.error(
                        f"Perplexica API returned HTTP {response.status_code}"
                    )
                    raise RuntimeError(
                        f"Perplexica API returned HTTP {response.status_code}: {response.text}"
                    )

                response_data = response.json()
                results = self._parse_perplexica_response(response_data, query)
                logfire.info(f"Perplexica search complete: {len(results)} results")
                return results

            except httpx.TimeoutException:  # type: ignore[misc]
                logfire.error(f"Perplexica search timed out after {self.timeout}s")
                raise RuntimeError(f"Perplexica search timed out after {self.timeout}s")
            except httpx.ConnectError:  # type: ignore[misc]
                logfire.error(f"Could not connect to Perplexica at {self.base_url}")
                raise RuntimeError(
                    f"Could not connect to Perplexica at {self.base_url}"
                )
            except httpx.RequestError as e:  # type: ignore[misc]
                logfire.error(f"Perplexica network error: {str(e)}")
                raise RuntimeError(f"Perplexica network error: {str(e)}")
            except RuntimeError:
                # Re-raise runtime errors
                raise
            except Exception as e:
                # Wrap unexpected errors
                logfire.error(f"Unexpected error during Perplexica search: {str(e)}")
                raise RuntimeError(
                    f"Unexpected error during Perplexica search: {str(e)}"
                )

    async def _async_health_check(self) -> bool:
        """Check if Perplexica API is available and working (async)."""
        try:
            session = await self._get_session()
            # Try to get available models as a health check
            response = await session.get(f"{self.base_url}/api/models", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def healthy(self) -> bool:
        """Check if Perplexica API is available and working."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context, assume healthy
                return True
            else:
                return loop.run_until_complete(self._async_health_check())
        except RuntimeError:
            # No event loop, create one
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
