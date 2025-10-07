"""
Perplexica search backend for AI-powered academic search.

This module provides a SearchBackend implementation for querying Perplexica,
an AI-powered search engine that can perform academic searches with context
understanding and real-time information retrieval.
"""

import asyncio
import re
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse

import httpx

from ..base import (
    SearchBackend,
    SearchQuery,
    SearchResults,
    SearchResult,
    SearchError,
    SearchTimeoutError,
    SearchUnavailableError,
)


class PerplexicaBackend(SearchBackend):
    """Perplexica search backend using local API instance."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Perplexica backend with configuration."""
        super().__init__(config)

        self.base_url = config.get("base_url", "http://localhost:3000")
        self.timeout = config.get("timeout", 60)
        self.search_mode = config.get("search_mode", "webSearch")
        self.max_sources = config.get("max_sources", 20)

        # Model configurations
        self.chat_model = config.get(
            "chat_model", {"provider": "openai", "name": "gpt-4o-mini"}
        )
        self.embedding_model = config.get(
            "embedding_model", {"provider": "openai", "name": "text-embedding-3-large"}
        )

        # Session for connection reuse
        self._session: Optional[httpx.AsyncClient] = None

    @property
    def backend_name(self) -> str:
        """Name identifier for this backend."""
        return "perplexica"

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session."""
        if self._session is None or self._session.is_closed:
            self._session = httpx.AsyncClient(timeout=self.timeout)
        return self._session

    def _build_search_request(self, query: SearchQuery) -> Dict[str, Any]:
        """Build the request payload for Perplexica API."""
        request_data = {
            "chatModel": self.chat_model,
            "embeddingModel": self.embedding_model,
            "optimizationMode": "balanced",  # Balance speed and quality
            "focusMode": self.search_mode,
            "query": query.full_query,
            "stream": False,  # Use non-streaming for simpler processing
        }

        # Add system instructions only for academic searches
        if self.search_mode == "academicSearch":
            request_data["systemInstructions"] = (
                "Focus on peer-reviewed scientific literature, research papers, and academic sources."
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

        # Don't extract anything - just use what Perplexica provides

        # Use page content as abstract - no filtering
        abstract = page_content if page_content else None

        # Calculate a simple relevance score based on position (earlier = more relevant)
        relevance_score = max(0.1, 1.0 - (index * 0.05))

        return SearchResult(
            title=title,
            url=url,
            snippet=abstract,
            relevance_score=relevance_score,
            backend=self.backend_name,
            metadata={
                "perplexica_source": source,
                "page_content_length": len(page_content),
            },
        )

    async def search(self, query: SearchQuery) -> SearchResults:
        """Perform a search with the given query."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Build request payload
            request_data = self._build_search_request(query)

            # Make API request
            session = await self._get_session()
            url = f"{self.base_url}/api/search"

            response = await session.post(url, json=request_data)

            if response.status_code == 400:
                raise SearchError(
                    f"Bad request to Perplexica: {response.text}",
                    backend=self.backend_name,
                    query=query.query,
                )
            elif response.status_code == 500:
                raise SearchError(
                    "Perplexica internal server error",
                    backend=self.backend_name,
                    query=query.query,
                )
            elif response.status_code != 200:
                raise SearchError(
                    f"HTTP {response.status_code}: {response.text}",
                    backend=self.backend_name,
                    query=query.query,
                )

            response_data = response.json()

            # Parse results
            results = self._parse_perplexica_response(response_data, query)
            search_time = asyncio.get_event_loop().time() - start_time

            # Perplexica doesn't provide total count, use results count
            total_found = len(response_data.get("sources", []))

            return SearchResults(
                query=query,
                results=results,
                total_found=total_found,
                search_time=search_time,
                backend=self.backend_name,
            )

        except httpx.TimeoutException:
            raise SearchTimeoutError(
                f"Perplexica search timed out after {self.timeout}s",
                backend=self.backend_name,
                query=query.query,
            )
        except httpx.ConnectError:
            raise SearchUnavailableError(
                f"Could not connect to Perplexica at {self.base_url}",
                backend=self.backend_name,
                query=query.query,
            )
        except httpx.RequestError as e:
            raise SearchError(
                f"Network error: {str(e)}", backend=self.backend_name, query=query.query
            )
        except SearchError:
            # Re-raise search errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise SearchError(
                f"Unexpected error during Perplexica search: {str(e)}",
                backend=self.backend_name,
                query=query.query,
            )

    async def health_check(self) -> bool:
        """Check if Perplexica API is available and working."""
        try:
            session = await self._get_session()

            # Try to get available models as a health check
            response = await session.get(f"{self.base_url}/api/models", timeout=10)
            return response.status_code == 200

        except Exception:
            return False

    def supports_filters(self) -> List[str]:
        """Return list of supported filter types for Perplexica."""
        return [
            "date_range",
            "publication_type",
        ]  # Limited filtering compared to PubMed

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Return rate limiting information for Perplexica."""
        return {
            "requests_per_second": None,  # Depends on local setup
            "requests_per_minute": None,
            "requests_per_hour": None,
            "burst_limit": None,
            "notes": "Rate limits depend on local Perplexica instance and underlying LLM providers",
        }

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
