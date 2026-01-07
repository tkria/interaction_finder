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

from interaction_finder.logging import logfire, get_logger

logger = get_logger(__name__)
from ..models import (
    SearchBackend,
    SearchQuery,
    SearchResult,
)


class PerplexicaBackend(SearchBackend):
    """Perplexica search backend using local API instance."""

    VALID_SOURCES = ("web", "discussions", "academic")

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
        # Sources: list of source types e.g. ["web"], ["academic"], ["web", "academic"]
        sources = config.get("sources", ["web"])
        self.sources: List[str] = sources if isinstance(sources, list) else [sources]
        for src in self.sources:
            if src not in self.VALID_SOURCES:
                raise ValueError(
                    f"Invalid source '{src}'. Valid sources: {self.VALID_SOURCES}"
                )
        # Optimization mode: speed (2 iterations), balanced (6), quality (25)
        self.optimization_mode = config.get("optimization_mode", "balanced")
        # Model configurations: {providerId: <provider-name>, key: <model-key>}
        # If not provided, will be fetched from /api/providers on first search
        self._chat_model = config.get("chat_model")
        self._embedding_model = config.get("embedding_model")
        self._models_initialized = (
            self._chat_model is not None and self._embedding_model is not None
        )

        # Session for connection reuse
        self._session: Optional[httpx.AsyncClient] = None  # type: ignore[valid-type]

    # Preferred models (first available wins, then falls back to any available)
    _PREFERRED_CHAT_MODELS = [
        "gpt-5-mini",
        "gpt-5",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4",
        "gpt-3.5-turbo",
    ]
    _PREFERRED_EMBEDDING_MODELS = [
        "text-embedding-3-large",
        "text-embedding-3-small",
        "Xenova/all-MiniLM-L6-v2",
    ]

    async def _fetch_default_models(self) -> None:
        """Fetch default models from Perplexica's /api/providers endpoint."""
        if self._models_initialized:
            return
        session = await self._get_session()
        try:
            response = await session.get(f"{self.base_url}/api/providers", timeout=10)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to fetch providers from Perplexica: HTTP {response.status_code}"
                )
            providers = response.json().get("providers", [])
            # Build {model_key: provider_id} lookups (provider_id is a UUID)
            chat_models: Dict[str, str] = {}
            embedding_models: Dict[str, str] = {}
            for p in providers:
                provider_id = p.get("id")
                if not provider_id:
                    continue
                for m in p.get("chatModels", []):
                    if key := m.get("key"):
                        chat_models.setdefault(key, provider_id)
                for m in p.get("embeddingModels", []):
                    if key := m.get("key"):
                        embedding_models.setdefault(key, provider_id)

            # Select preferred model or first available
            def select(
                available: Dict[str, str], preferred: List[str], kind: str
            ) -> Dict[str, str]:
                for key in preferred:
                    if key in available:
                        return {"providerId": available[key], "key": key}
                if available:
                    key, provider = next(iter(available.items()))
                    return {"providerId": provider, "key": key}
                raise RuntimeError(
                    f"No {kind} models available in Perplexica providers."
                )

            self._chat_model = select(chat_models, self._PREFERRED_CHAT_MODELS, "chat")
            self._embedding_model = select(
                embedding_models, self._PREFERRED_EMBEDDING_MODELS, "embedding"
            )
            self._models_initialized = True
            logger.info(
                "Auto-configured Perplexica models",
                chat_model=self._chat_model,
                embedding_model=self._embedding_model,
            )
        except httpx.RequestError as e:  # type: ignore[misc]
            raise RuntimeError(f"Failed to connect to Perplexica: {e}")

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
        """Build the request payload for Perplexica API.

        Note: _fetch_default_models() must be called before this method
        if models were not provided in config.
        """
        return {
            "optimizationMode": self.optimization_mode,
            "sources": self.sources,
            "query": query.query,
            "stream": False,
            "chatModel": self._chat_model,
            "embeddingModel": self._embedding_model,
            "history": [],
        }

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
        query_text = query.query
        with logfire.span(
            f"Perplexica: {query_text}",
            query=query_text,
            max_results=query.max_results,
        ):
            try:
                # Ensure models are configured (fetches from /api/providers if needed)
                await self._fetch_default_models()
                request_data = self._build_search_request(query)
                session = await self._get_session()
                url = f"{self.base_url}/api/search"

                response = await session.post(url, json=request_data)

                # Handle HTTP errors
                if response.status_code == 500:
                    logger.error("Perplexica internal server error")
                    raise RuntimeError("Perplexica internal server error")
                elif response.status_code != 200:
                    logger.error(f"Perplexica API returned HTTP {response.status_code}")
                    raise RuntimeError(
                        f"Perplexica API returned HTTP {response.status_code}: {response.text}"
                    )

                response_data = response.json()
                results = self._parse_perplexica_response(response_data, query)
                logger.info(
                    f"Perplexica search returned {len(results)} results",
                    extra={
                        "query": query_text,
                        "results": results,
                    },
                )
                return results

            except httpx.TimeoutException:  # type: ignore[misc]
                logger.error(f"Perplexica search timed out after {self.timeout}s")
                raise RuntimeError(f"Perplexica search timed out after {self.timeout}s")
            except httpx.ConnectError:  # type: ignore[misc]
                logger.error(f"Could not connect to Perplexica at {self.base_url}")
                raise RuntimeError(
                    f"Could not connect to Perplexica at {self.base_url}"
                )
            except httpx.RequestError as e:  # type: ignore[misc]
                logger.error(f"Perplexica network error: {str(e)}")
                raise RuntimeError(f"Perplexica network error: {str(e)}")
            except RuntimeError:
                # Re-raise runtime errors
                raise
            except Exception as e:
                # Wrap unexpected errors
                logger.error(f"Unexpected error during Perplexica search: {str(e)}")
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
        except Exception as exc:
            logger.warning(
                "Perplexica health check failed",
                extra={
                    "error": str(exc),
                    "base_url": self.base_url,
                },
            )
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
