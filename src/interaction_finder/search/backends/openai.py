"""
OpenAI web search backend using the Responses API.

This module provides a simplified SearchBackend implementation for OpenAI's
web search tool using the Responses API with structured search results.
"""

import asyncio
import os
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore

from interaction_finder.logging import logfire, get_logger

logger = get_logger(__name__)
from ..models import (
    SearchBackend,
    SearchQuery,
    SearchResult,
)


class OpenAIBackend(SearchBackend):
    """OpenAI web search backend using Responses API."""

    def __init__(self, config: Dict[str, Any] = {}):
        """Initialize OpenAI backend with configuration."""
        super().__init__(config)

        if not HTTPX_AVAILABLE:
            raise RuntimeError(
                "httpx is required for OpenAI backend but not installed. "
                "Install it with: pip install httpx"
            )

        # Get API key from config or environment
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment "
                "variable or configure api_key in config file."
            )

        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.model = config.get("model", "gpt-4o-mini")
        self.timeout = config.get("timeout", 60)

        # Session for connection reuse
        self._session: Optional[httpx.AsyncClient] = None  # type: ignore[valid-type]

    @property
    def name(self) -> str:
        """Name identifier for this backend."""
        return "openai"

    async def _get_session(self) -> httpx.AsyncClient:  # type: ignore[valid-type]
        """Get or create HTTP session."""
        if self._session is None or self._session.is_closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            self._session = httpx.AsyncClient(  # type: ignore[misc]
                headers=headers, timeout=self.timeout
            )
        return self._session

    def _build_search_request(self, query: SearchQuery) -> Dict[str, Any]:
        """Build the request payload for OpenAI Responses API."""
        return {
            "model": self.model,
            "tools": [{"type": "web_search"}],
            "tool_choice": "required",
            "include": ["web_search_call.action.sources"],
            "input": f"Please search the web for information about: {query.query}",
        }

    def _clean_url(self, url: str) -> str:
        """Remove tracking parameters from URLs."""
        if not url:
            return url

        parsed = urlparse(url)
        if not parsed.query:
            return url

        # Parse and filter tracking parameters
        params = parse_qs(parsed.query, keep_blank_values=False)
        tracking_params = {
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_term",
            "utm_content",
            "fbclid",
            "gclid",
            "msclkid",
        }
        clean_params = {k: v for k, v in params.items() if k not in tracking_params}

        # Reconstruct URL
        clean_query = urlencode(clean_params, doseq=True) if clean_params else ""
        return urlunparse(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                clean_query,
                parsed.fragment,
            )
        )

    def _parse_openai_response(
        self, response_data: Dict[str, Any] | list, query: SearchQuery
    ) -> List[SearchResult]:
        """Parse OpenAI Responses API output into SearchResult objects.

        The response_data can be either:
        - A dict with 'outputs' or 'output' key (standard API response)
        - A list directly (alternative API format or nested extraction)

        This flexibility handles both documented and observed API behaviors.
        """
        results = []

        # Extract outputs - handle both list format and nested keys
        if isinstance(response_data, list):
            outputs = response_data
        else:
            outputs = response_data.get("outputs", response_data.get("output", []))

        # Find web search call and message outputs
        web_search_call = None
        message_output = None
        for output in outputs:
            if output.get("type") == "web_search_call":
                web_search_call = output
            elif output.get("type") == "message":
                message_output = output

        if not web_search_call or not message_output:
            return results

        # Build URL to citation mapping from message content
        citations_by_url = {}
        content = message_output.get("content", [])
        for content_item in content:
            if content_item.get("type") == "output_text":
                annotations = content_item.get("annotations", [])
                for annotation in annotations:
                    if annotation.get("type") == "url_citation":
                        citation_url = annotation.get("url", "")
                        if citation_url:
                            citations_by_url[citation_url] = annotation
                            # Also map cleaned URL
                            cleaned = self._clean_url(citation_url)
                            if cleaned != citation_url:
                                citations_by_url[cleaned] = annotation

        # Create SearchResult objects from citations (have titles)
        processed_urls = set()
        for i, annotation in enumerate(citations_by_url.values()):
            if len(results) >= query.max_results:
                break

            citation_url = annotation.get("url", "")
            if not citation_url or citation_url in processed_urls:
                continue

            processed_urls.add(citation_url)
            clean_url = self._clean_url(citation_url)

            # Use citation title or domain as fallback
            title = annotation.get("title", "")
            if not title:
                parsed_url = urlparse(clean_url)
                title = parsed_url.netloc

            # Simple relevance score based on position
            relevance = max(0.1, 1.0 - (i * 0.05))

            results.append(
                SearchResult(
                    title=title,
                    url=clean_url,
                    snippet=None,
                    relevance=relevance,
                )
            )

        # Fall back to sources without citations if needed
        if len(results) < query.max_results:
            action = web_search_call.get("action", {})
            sources = action.get("sources", [])

            for source in sources:
                if len(results) >= query.max_results:
                    break

                # Extract URL from source
                if isinstance(source, dict):
                    source_url = source.get("url", "")
                else:
                    source_url = str(source)

                if not source_url:
                    continue

                clean_url = self._clean_url(source_url)
                if clean_url in processed_urls:
                    continue

                processed_urls.add(clean_url)

                # Use domain as title
                parsed_url = urlparse(clean_url)
                title = parsed_url.netloc

                relevance = max(0.1, 1.0 - (len(results) * 0.05))

                results.append(
                    SearchResult(
                        title=title,
                        url=clean_url,
                        snippet=None,
                        relevance=relevance,
                    )
                )

        return results

    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform a search with the given query."""
        query_text = query.query
        with logfire.span(
            f"OpenAI: {query_text}",
            query=query_text,
            max_results=query.max_results,
            model=self.model,
        ):
            try:
                request_data = self._build_search_request(query)
                session = await self._get_session()
                url = f"{self.base_url}/responses"

                response = await session.post(url, json=request_data)

                # Handle HTTP errors
                if response.status_code == 401:
                    logger.error("OpenAI API authentication failed")
                    raise RuntimeError(
                        "OpenAI API authentication failed - check API key"
                    )
                elif response.status_code == 429:
                    logger.error("OpenAI API rate limit exceeded")
                    raise RuntimeError("OpenAI API rate limit exceeded")
                elif response.status_code != 200:
                    logger.error(f"OpenAI API returned HTTP {response.status_code}")
                    raise RuntimeError(
                        f"OpenAI API returned HTTP {response.status_code}: {response.text}"
                    )

                response_data = response.json()
                results = self._parse_openai_response(response_data, query)
                logger.info(
                    f"OpenAI search returned {len(results)} results",
                    extra={
                        "query": query_text,
                        "results": results,
                    },
                )
                return results

            except httpx.TimeoutException:  # type: ignore[misc]
                logger.error(f"OpenAI search timed out after {self.timeout}s")
                raise RuntimeError(f"OpenAI search timed out after {self.timeout}s")
            except httpx.RequestError as e:  # type: ignore[misc]
                logger.error(f"OpenAI network error: {str(e)}")
                raise RuntimeError(f"OpenAI network error: {str(e)}")
            except RuntimeError:
                # Re-raise runtime errors
                raise
            except Exception as e:
                # Wrap unexpected errors
                logger.error(f"Unexpected error during OpenAI search: {str(e)}")
                raise RuntimeError(f"Unexpected error during OpenAI search: {str(e)}")

    async def _async_health_check(self) -> bool:
        """Check if OpenAI API is available and working (async)."""
        try:
            session = await self._get_session()
            test_request = {
                "model": self.model,
                "tools": [{"type": "web_search"}],
                "input": "test",
            }
            response = await session.post(
                f"{self.base_url}/responses", json=test_request, timeout=10
            )
            # Both 200 and 400 indicate API is reachable
            return response.status_code in [200, 400]
        except Exception as exc:
            logger.warning(
                "OpenAI search backend health check failed",
                extra={
                    "error": str(exc),
                    "base_url": self.base_url,
                },
            )
            return False

    def healthy(self) -> bool:
        """Check if OpenAI API is available and working."""
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
