"""
OpenAI web search backend using the Responses API.

This module provides a SearchBackend implementation for OpenAI's web search
tool using the Responses API, which provides structured search results with
proper citations and source tracking.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

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


class OpenAISearchBackend(SearchBackend):
    """OpenAI web search backend using Responses API."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize OpenAI search backend with configuration."""
        super().__init__(config)

        self.api_key = config.get("api_key")
        if not self.api_key:
            # Try to get API key from environment variable
            import os

            self.api_key = os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required for OpenAI search backend. Set OPENAI_API_KEY environment variable or configure api_key in config file."
            )

        self.base_url = config.get("base_url", "https://api.openai.com/v1")
        self.timeout = config.get("timeout", 60)
        self.model = config.get("model", "gpt-4o-mini")
        self.reasoning_effort = config.get("reasoning_effort", "low")
        self.allowed_domains = config.get("allowed_domains", [])
        self.user_location = config.get("user_location", {})

        # Session for connection reuse
        self._session: Optional[httpx.AsyncClient] = None

    @property
    def backend_name(self) -> str:
        """Name identifier for this backend."""
        return "openai_search"

    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create HTTP session."""
        if self._session is None or self._session.is_closed:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            self._session = httpx.AsyncClient(headers=headers, timeout=self.timeout)
        return self._session

    def _build_search_request(self, query: SearchQuery) -> Dict[str, Any]:
        """Build the request payload for OpenAI Responses API."""
        # Build web search tool configuration
        web_search_tool = {"type": "web_search"}

        # Add domain filtering if configured
        if self.allowed_domains:
            web_search_tool["filters"] = {
                "allowed_domains": self.allowed_domains[:20]  # Max 20 domains
            }

        # Add user location if configured
        if self.user_location:
            web_search_tool["user_location"] = self.user_location

        request_data = {
            "model": self.model,
            "tools": [web_search_tool],
            "tool_choice": "required",  # Force tool usage
            "include": ["web_search_call.action.sources"],
            "input": f"Please search the web for information about: {query.full_query}",
        }

        # Add reasoning configuration for supported models
        if self.model in ["gpt-5", "o3", "o4-mini"] and self.reasoning_effort:
            request_data["reasoning"] = {"effort": self.reasoning_effort}

        return request_data

    def _clean_url(self, url: str) -> str:
        """Remove UTM and other tracking parameters from URLs."""
        if not url:
            return url

        parsed = urlparse(url)
        if not parsed.query:
            return url

        # Parse query parameters
        params = parse_qs(parsed.query, keep_blank_values=False)

        # Remove UTM and other tracking parameters
        tracking_params = {
            "utm_source",
            "utm_medium",
            "utm_campaign",
            "utm_term",
            "utm_content",
            "fbclid",
            "gclid",
            "msclkid",
            "mc_eid",
            "mc_cid",
            "_ga",
            "_gl",
            "_hsenc",
            "_hsmi",
        }

        # Filter out tracking parameters
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
        self, response_data: Dict[str, Any], query: SearchQuery
    ) -> List[SearchResult]:
        """Parse OpenAI Responses API output into SearchResult objects."""
        results = []

        # Look for web search call and message outputs
        web_search_call = None
        message_output = None

        # Handle both list format and nested 'outputs'/'output' keys
        if isinstance(response_data, list):
            outputs = response_data
        else:
            outputs = response_data.get("outputs", response_data.get("output", []))

        for output in outputs:
            if output.get("type") == "web_search_call":
                web_search_call = output
            elif output.get("type") == "message":
                message_output = output

        if not web_search_call:
            # If no web search was performed, return empty results
            return results

        if not message_output:
            return results

        # Extract citations from message content - create URL -> citation mapping
        citations_by_url = {}
        content = message_output.get("content", [])
        for content_item in content:
            if content_item.get("type") == "output_text":
                annotations = content_item.get("annotations", [])
                for annotation in annotations:
                    if annotation.get("type") == "url_citation":
                        citation_url = annotation.get("url", "")
                        if citation_url:
                            # Store both cleaned and original URLs as keys
                            citations_by_url[citation_url] = annotation
                            cleaned_url = self._clean_url(citation_url)
                            if cleaned_url != citation_url:
                                citations_by_url[cleaned_url] = annotation

        # Create SearchResult objects from citations (citations have titles)
        processed_urls = set()

        for i, annotation in enumerate(citations_by_url.values()):
            if len(results) >= query.max_results:
                break

            citation_url = annotation.get("url", "")
            citation_title = annotation.get("title", "")

            if not citation_url or citation_url in processed_urls:
                continue

            processed_urls.add(citation_url)

            # Clean URL to remove tracking parameters
            clean_url = self._clean_url(citation_url)

            # Use citation title if available, otherwise fall back to domain
            if citation_title:
                title = citation_title
            else:
                parsed_url = urlparse(clean_url)
                title = parsed_url.netloc

            # Calculate relevance score based on position
            relevance_score = max(0.1, 1.0 - (i * 0.05))

            result = SearchResult(
                title=title,
                url=clean_url,
                relevance_score=relevance_score,
                backend=self.backend_name,
                metadata={
                    "citation": annotation,
                    "search_call_id": web_search_call.get("id"),
                },
            )
            results.append(result)

        # If we still don't have enough results, fall back to sources without citations
        if len(results) < query.max_results:
            sources = []
            action = web_search_call.get("action", {})
            if "sources" in action:
                sources = action["sources"]

            for source in sources:
                if len(results) >= query.max_results:
                    break

                # Extract URL from source object
                if isinstance(source, dict):
                    source_url = source.get("url", "")
                else:
                    source_url = str(source)

                if not source_url:
                    continue

                # Clean URL to remove tracking parameters
                clean_url = self._clean_url(source_url)

                # Skip if we already processed this URL
                if clean_url in processed_urls:
                    continue

                processed_urls.add(clean_url)

                # Use domain name as title for sources without citations
                parsed_url = urlparse(clean_url)
                title = parsed_url.netloc

                # Calculate relevance score based on position
                relevance_score = max(0.1, 1.0 - (len(results) * 0.05))

                result = SearchResult(
                    title=title,
                    url=clean_url,
                    relevance_score=relevance_score,
                    backend=self.backend_name,
                    metadata={
                        "source": source,
                        "search_call_id": web_search_call.get("id"),
                    },
                )
                results.append(result)

        return results

    async def search(self, query: SearchQuery) -> SearchResults:
        """Perform a search with the given query."""
        start_time = asyncio.get_event_loop().time()

        try:
            # Build request payload
            request_data = self._build_search_request(query)

            # Make API request
            session = await self._get_session()
            url = f"{self.base_url}/responses"

            response = await session.post(url, json=request_data)

            if response.status_code == 400:
                error_data = response.json() if response.content else {}
                error_message = error_data.get("error", {}).get(
                    "message", response.text
                )
                raise SearchError(
                    f"Bad request to OpenAI: {error_message}",
                    backend=self.backend_name,
                    query=query.query,
                )
            elif response.status_code == 401:
                raise SearchError(
                    "OpenAI API authentication failed - check API key",
                    backend=self.backend_name,
                    query=query.query,
                )
            elif response.status_code == 403:
                raise SearchError(
                    "OpenAI API access forbidden - check permissions",
                    backend=self.backend_name,
                    query=query.query,
                )
            elif response.status_code == 429:
                raise SearchError(
                    "OpenAI API rate limit exceeded",
                    backend=self.backend_name,
                    query=query.query,
                )
            elif response.status_code == 500:
                raise SearchError(
                    "OpenAI internal server error",
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
            results = self._parse_openai_response(response_data, query)
            search_time = asyncio.get_event_loop().time() - start_time

            # OpenAI doesn't provide total count, use results count
            total_found = len(results)

            return SearchResults(
                query=query,
                results=results,
                total_found=total_found,
                search_time=search_time,
                backend=self.backend_name,
            )

        except httpx.TimeoutException:
            raise SearchTimeoutError(
                f"OpenAI search timed out after {self.timeout}s",
                backend=self.backend_name,
                query=query.query,
            )
        except httpx.ConnectError:
            raise SearchUnavailableError(
                f"Could not connect to OpenAI API at {self.base_url}",
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
                f"Unexpected error during OpenAI search: {str(e)}",
                backend=self.backend_name,
                query=query.query,
            )

    async def health_check(self) -> bool:
        """Check if OpenAI API is available and working."""
        try:
            session = await self._get_session()

            # Try a minimal request to check API access
            test_request = {
                "model": self.model,
                "tools": [{"type": "web_search"}],
                "input": "test",
            }

            response = await session.post(
                f"{self.base_url}/responses", json=test_request, timeout=10
            )
            return response.status_code in [200, 400]  # 400 is OK (bad request format)

        except Exception:
            return False

    def supports_filters(self) -> List[str]:
        """Return list of supported filter types for OpenAI search."""
        return [
            "allowed_domains",
            "user_location",
        ]

    def get_rate_limit_info(self) -> Dict[str, Any]:
        """Return rate limiting information for OpenAI search."""
        return {
            "requests_per_second": None,  # Depends on OpenAI tier
            "requests_per_minute": None,
            "requests_per_hour": None,
            "burst_limit": None,
            "notes": "Rate limits depend on OpenAI API tier and model used",
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
