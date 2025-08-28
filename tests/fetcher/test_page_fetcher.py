"""
Tests for the PageFetcher high-level interface.

Tests cover:
- PageFetcher initialization and configuration
- get_chunks_with_embeddings functionality
- get_chunks text extraction
- Error handling and edge cases
- Integration with WebClient and URLCache
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from interaction_finder.fetcher.page_fetcher import PageFetcher
from interaction_finder.settings import IfetcherConfig


class TestPageFetcherInitialization:
    """Test PageFetcher initialization and configuration."""

    def test_pagefetcher_creation_basic(self):
        """Test basic PageFetcher creation."""
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "test_cache"
        config.abspath.return_value = Path("test_cache")

        fetcher = PageFetcher(config)

        assert fetcher.config == config
        assert fetcher.verbose is False  # Default
        assert fetcher.web_client is not None
        assert fetcher.cache is not None

    def test_pagefetcher_creation_with_options(self):
        """Test PageFetcher creation with custom options."""
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "custom_cache"
        config.abspath.return_value = Path("custom_cache")

        fetcher = PageFetcher(config, show_status=False)

        assert fetcher.verbose is False  # show_status is passed to batch_ops, not stored on fetcher

    def test_pagefetcher_abspath_delegation(self):
        """Test that abspath is properly delegated to config."""
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "cache"

        # Create a temporary directory for the test instead of using /absolute/path
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "cache"
            config.abspath.return_value = temp_path

            fetcher = PageFetcher(config)
            result = fetcher.config.abspath("relative/path", param="value")

            # config.abspath is called twice: once during URLCache init with "cache",
            # and once by our test call with "relative/path"
            assert config.abspath.call_count == 2
            config.abspath.assert_any_call("cache")
            config.abspath.assert_any_call("relative/path", param="value")
            assert result == temp_path


class TestGetChunksWithEmbeddings:
    """Test get_chunks_with_embeddings functionality."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "cache"
        config.abspath.return_value = tmp_path / "cache"
        return config

    @pytest.fixture
    def fetcher(self, mock_config):
        return PageFetcher(mock_config, show_status=False)

    @pytest.mark.asyncio
    async def test_get_chunks_with_embeddings_cache_hit(self, fetcher):
        """Test getting chunks when data is in cache."""
        url = "http://example.com/test"
        cached_data = {
            "chunks": [
                {"text": "chunk1", "wordcount": 5, "embedding": [1.0, 2.0, 3.0]},
                {"text": "chunk2", "wordcount": 7, "embedding": [4.0, 5.0, 6.0]},
            ]
        }

        # Mock cache to return data
        fetcher.cache.has_path = AsyncMock(return_value=True)
        fetcher.cache.get_content = AsyncMock(return_value=cached_data)

        result = await fetcher.get_chunks_with_embeddings(url)

        assert len(result) == 2
        assert result[0]["text"] == "chunk1"
        assert result[1]["text"] == "chunk2"
        fetcher.cache.has_path.assert_called_once_with(url, "chunks")
        fetcher.cache.get_content.assert_called_once_with(url, "chunks")

    @pytest.mark.asyncio
    async def test_get_chunks_with_embeddings_cache_miss(self, fetcher):
        """Test getting chunks when not in cache (requires web fetching)."""
        url = "http://example.com/test"

        # Mock cache miss
        fetcher.cache.has_path = AsyncMock(return_value=False)

        # Mock web client
        mock_markdown = "Test markdown content."
        mock_chunks = [
            {"text": "chunk1", "wordcount": 5, "embedding": [1.0, 2.0, 3.0]},
        ]

        fetcher.get_markdown = AsyncMock(return_value=mock_markdown)
        fetcher.web_client.create_chunks = Mock(return_value=mock_chunks)
        fetcher.cache.set_content = AsyncMock()

        result = await fetcher.get_chunks_with_embeddings(url)

        # Check result
        assert len(result) == 1
        assert result[0]["text"] == "chunk1"

        # Check calls
        fetcher.get_markdown.assert_called_once_with(url, retry=False)
        fetcher.web_client.create_chunks.assert_called_once_with(mock_markdown)

        # Should cache the result
        fetcher.cache.set_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_chunks_with_embeddings_retry(self, fetcher):
        """Test retry functionality."""
        url = "http://example.com/test"

        fetcher.cache.has_path = AsyncMock(return_value=False)
        fetcher.get_markdown = AsyncMock(return_value="content")
        fetcher.web_client.create_chunks = Mock(return_value=[])
        fetcher.cache.set_content = AsyncMock()

        await fetcher.get_chunks_with_embeddings(url, retry=True)

        fetcher.get_markdown.assert_called_once_with(url, retry=True)


class TestGetChunks:
    """Test get_chunks text extraction functionality."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "cache"
        config.abspath.return_value = tmp_path / "cache"
        return config

    @pytest.fixture
    def fetcher(self, mock_config):
        return PageFetcher(mock_config, show_status=False)

    @pytest.mark.asyncio
    async def test_get_chunks_extracts_text(self, fetcher):
        """Test that get_chunks extracts text from chunks with embeddings."""
        chunks_with_embeddings = [
            {"text": "First chunk text", "wordcount": 3, "embedding": [1.0, 2.0]},
            {"text": "Second chunk text", "wordcount": 3, "embedding": [3.0, 4.0]},
        ]

        fetcher.get_chunks_with_embeddings = AsyncMock(
            return_value=chunks_with_embeddings
        )

        result = await fetcher.get_chunks("http://example.com/test")

        assert result == ["First chunk text", "Second chunk text"]
        fetcher.get_chunks_with_embeddings.assert_called_once_with(
            "http://example.com/test", retry=False
        )

    @pytest.mark.asyncio
    async def test_get_chunks_empty_result(self, fetcher):
        """Test get_chunks with empty chunks."""
        fetcher.get_chunks_with_embeddings = AsyncMock(return_value=[])

        result = await fetcher.get_chunks("http://example.com/test")

        assert result == []

    @pytest.mark.asyncio
    async def test_get_chunks_with_retry(self, fetcher):
        """Test get_chunks retry parameter."""
        fetcher.get_chunks_with_embeddings = AsyncMock(return_value=[])

        await fetcher.get_chunks("http://example.com/test", retry=True)

        fetcher.get_chunks_with_embeddings.assert_called_once_with(
            "http://example.com/test", retry=True
        )


class TestDocumentEmbeddings:
    """Test document embedding computation."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "cache"
        config.abspath.return_value = tmp_path / "cache"
        return config

    @pytest.fixture
    def fetcher(self, mock_config):
        return PageFetcher(mock_config, show_status=False)

    def test_compute_document_embeddings_single_doc(self, fetcher):
        """Test document embedding computation for single document."""
        doc_chunks = {
            "doc1": [
                {"embedding": [1.0, 2.0, 3.0]},
                {"embedding": [4.0, 5.0, 6.0]},
            ]
        }

        result = fetcher._compute_document_embeddings(doc_chunks)

        assert "doc1" in result
        # Should be average: [(1+4)/2, (2+5)/2, (3+6)/2] = [2.5, 3.5, 4.5]
        import numpy as np
        assert isinstance(result["doc1"], np.ndarray)
        np.testing.assert_array_almost_equal(result["doc1"], [2.5, 3.5, 4.5])

    def test_compute_document_embeddings_multiple_docs(self, fetcher):
        """Test document embedding computation for multiple documents."""
        doc_chunks = {
            "doc1": [{"embedding": [1.0, 2.0]}],
            "doc2": [{"embedding": [3.0, 4.0]}, {"embedding": [5.0, 6.0]}],
        }

        result = fetcher._compute_document_embeddings(doc_chunks)

        import numpy as np
        assert isinstance(result["doc1"], np.ndarray)
        assert isinstance(result["doc2"], np.ndarray)
        np.testing.assert_array_almost_equal(result["doc1"], [1.0, 2.0])
        np.testing.assert_array_almost_equal(result["doc2"], [4.0, 5.0])  # Average of [3,4] and [5,6]

    def test_compute_document_embeddings_no_embeddings(self, fetcher):
        """Test document embedding with chunks that have no embeddings."""
        doc_chunks = {
            "doc1": [{"text": "chunk without embedding"}],
        }

        result = fetcher._compute_document_embeddings(doc_chunks)

        # Should skip documents without embeddings
        assert result == {}

    def test_compute_document_embeddings_mixed(self, fetcher):
        """Test with mix of chunks with and without embeddings."""
        doc_chunks = {
            "doc1": [
                {"embedding": [1.0, 2.0]},
                {"text": "no embedding"},
                {"embedding": [3.0, 4.0]},
            ],
        }

        result = fetcher._compute_document_embeddings(doc_chunks)

        # Should average only the chunks with embeddings
        import numpy as np
        assert isinstance(result["doc1"], np.ndarray)
        np.testing.assert_array_almost_equal(result["doc1"], [2.0, 3.0])  # Average of [1,2] and [3,4]


class TestErrorHandling:
    """Test error handling in PageFetcher."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "cache"
        config.abspath.return_value = tmp_path / "cache"
        return config

    @pytest.fixture
    def fetcher(self, mock_config):
        return PageFetcher(mock_config, show_status=False)

    @pytest.mark.asyncio
    async def test_get_chunks_web_client_error(self, fetcher):
        """Test error handling when web client fails."""
        fetcher.cache.has_path = AsyncMock(return_value=False)
        fetcher.get_markdown = AsyncMock(
            side_effect=RuntimeError("Network error")
        )

        with pytest.raises(RuntimeError, match="Network error"):
            await fetcher.get_chunks_with_embeddings("http://example.com/test")

    @pytest.mark.asyncio
    async def test_get_chunks_cache_error(self, fetcher):
        """Test handling of cache errors."""
        fetcher.cache.has_path = AsyncMock(side_effect=RuntimeError("Cache error"))

        with pytest.raises(RuntimeError, match="Cache error"):
            await fetcher.get_chunks_with_embeddings("http://example.com/test")


class TestIntegration:
    """Test integration between PageFetcher components."""

    @pytest.fixture
    def mock_config(self, tmp_path):
        config = Mock(spec=IfetcherConfig)
        config.output = Mock()
        config.output.cache = "cache"
        config.abspath.return_value = tmp_path / "cache"
        return config

    def test_pagefetcher_creates_web_client(self, mock_config):
        """Test that PageFetcher creates WebClient with correct config."""
        with patch(
            "interaction_finder.fetcher.page_fetcher.WebClient"
        ) as mock_web_client_class:
            mock_web_client_instance = Mock()
            mock_web_client_class.return_value = mock_web_client_instance

            fetcher = PageFetcher(mock_config, show_status=False)

            # Should create WebClient with config and verbose=False (as positional args)
            mock_web_client_class.assert_called_once_with(mock_config, False)
            assert fetcher.web_client == mock_web_client_instance

    def test_pagefetcher_creates_cache(self, mock_config):
        """Test that PageFetcher creates URLCache."""
        with patch(
            "interaction_finder.fetcher.page_fetcher.URLCache"
        ) as mock_cache_class:
            mock_cache_instance = Mock()
            mock_cache_class.return_value = mock_cache_instance

            fetcher = PageFetcher(mock_config, show_status=False)

            # Should create URLCache with cache path from config
            mock_cache_class.assert_called_once()
            assert fetcher.cache == mock_cache_instance
