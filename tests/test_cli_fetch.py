"""Tests for cli_fetch module - URL input parsing and validation."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from interaction_finder.cli_fetch import (
    read_urls_from_file,
    collect_urls,
    FetchResult,
    fetch_urls_markdown,
    fetch_urls_chunks,
    run_fetch,
)


class TestReadUrlsFromFile:
    """Tests for read_urls_from_file function."""

    def test_read_urls_basic(self, tmp_path):
        """Read basic URL list from file."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://example.com/1\nhttps://example.com/2\n")

        urls = read_urls_from_file(url_file)

        assert urls == ["https://example.com/1", "https://example.com/2"]

    def test_read_urls_strips_whitespace(self, tmp_path):
        """Strip leading and trailing whitespace from URLs."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "  https://example.com/1  \n"
            "\thttps://example.com/2\t\n"
            "https://example.com/3\n"
        )

        urls = read_urls_from_file(url_file)

        assert urls == [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ]

    def test_read_urls_skips_empty_lines(self, tmp_path):
        """Skip empty lines in URL file."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "https://example.com/1\n"
            "\n"
            "https://example.com/2\n"
            "  \n"
            "https://example.com/3\n"
        )

        urls = read_urls_from_file(url_file)

        assert urls == [
            "https://example.com/1",
            "https://example.com/2",
            "https://example.com/3",
        ]

    def test_read_urls_empty_file(self, tmp_path):
        """Empty file returns empty list."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text("")

        urls = read_urls_from_file(url_file)

        assert urls == []

    def test_read_urls_file_only_whitespace(self, tmp_path):
        """File with only whitespace returns empty list."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text("  \n\t\n  \n")

        urls = read_urls_from_file(url_file)

        assert urls == []

    def test_read_urls_file_not_found(self, tmp_path):
        """Raise FileNotFoundError for missing file."""
        missing_file = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError) as exc_info:
            read_urls_from_file(missing_file)

        # Error message should mention the missing file
        assert str(missing_file) in str(exc_info.value)

    def test_read_urls_utf8_encoding(self, tmp_path):
        """Handle UTF-8 encoded files."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://example.com/page?q=café\n", encoding="utf-8")

        urls = read_urls_from_file(url_file)

        assert urls == ["https://example.com/page?q=café"]


class TestCollectUrls:
    """Tests for collect_urls function."""

    def test_collect_from_args(self):
        """Collect URLs from command-line arguments."""
        urls = collect_urls(
            args_urls=["https://example.com/1", "https://example.com/2"],
            input_file=None,
        )

        assert urls == ["https://example.com/1", "https://example.com/2"]

    def test_collect_from_file(self, tmp_path):
        """Collect URLs from file input."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://example.com/1\nhttps://example.com/2\n")

        urls = collect_urls(args_urls=[], input_file=url_file)

        assert urls == ["https://example.com/1", "https://example.com/2"]

    def test_collect_mutual_exclusion_error(self, tmp_path):
        """Error when both args and file provided."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://example.com/1\n")

        with pytest.raises(ValueError) as exc_info:
            collect_urls(args_urls=["https://example.com/2"], input_file=url_file)

        # Error message should explain mutual exclusion
        error_msg = str(exc_info.value).lower()
        assert "both" in error_msg or "mutual" in error_msg

    def test_collect_empty_input_error(self):
        """Error when neither args nor file provided."""
        with pytest.raises(ValueError) as exc_info:
            collect_urls(args_urls=[], input_file=None)

        # Error message should explain no input provided
        error_msg = str(exc_info.value).lower()
        assert "no" in error_msg and ("url" in error_msg or "input" in error_msg)

    def test_collect_file_not_found_error(self, tmp_path):
        """FileNotFoundError when input file missing."""
        missing_file = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError) as exc_info:
            collect_urls(args_urls=[], input_file=missing_file)

        # Error should propagate from read_urls_from_file
        assert str(missing_file) in str(exc_info.value)

    def test_collect_empty_args_list(self):
        """Empty args list treated as no args."""
        with pytest.raises(ValueError) as exc_info:
            collect_urls(args_urls=[], input_file=None)

        assert "no" in str(exc_info.value).lower()

    def test_collect_none_args_treated_as_empty(self):
        """None args list treated as empty."""
        with pytest.raises(ValueError) as exc_info:
            collect_urls(args_urls=None, input_file=None)

        assert "no" in str(exc_info.value).lower()

    def test_collect_preserves_order(self, tmp_path):
        """URLs returned in original order."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text(
            "https://example.com/c\nhttps://example.com/a\nhttps://example.com/b\n"
        )

        urls = collect_urls(args_urls=[], input_file=url_file)

        # Order should be preserved (not sorted)
        assert urls == [
            "https://example.com/c",
            "https://example.com/a",
            "https://example.com/b",
        ]

    def test_collect_allows_duplicates(self):
        """Duplicates are preserved (deduplication handled elsewhere)."""
        urls = collect_urls(
            args_urls=[
                "https://example.com/1",
                "https://example.com/1",  # Duplicate
                "https://example.com/2",
            ],
            input_file=None,
        )

        # Duplicates should be preserved
        assert urls == [
            "https://example.com/1",
            "https://example.com/1",
            "https://example.com/2",
        ]


class TestFetchUrlsMarkdown:
    """Tests for fetch_urls_markdown function with mocked PageFetcher."""

    @pytest.mark.asyncio
    async def test_fetch_urls_markdown_success(self):
        """Successfully fetch multiple URLs and retrieve cache paths."""
        # Create mock fetcher with mocked fetch_documents
        fetcher = MagicMock()
        fetcher.fetch_documents = AsyncMock(
            return_value={
                "https://example.com/1": MagicMock(content_markdown="Content 1"),
                "https://example.com/2": MagicMock(content_markdown="Content 2"),
            }
        )
        # Mock cache.get_file_path to return Path objects
        fetcher.cache.get_file_path = AsyncMock(
            side_effect=lambda url, content_type: Path(
                f"/cache/{url.split('/')[-1]}.md"
            )
        )

        # Fetch URLs
        results = await fetch_urls_markdown(
            fetcher, ["https://example.com/1", "https://example.com/2"], verbose=False
        )

        # Verify results
        assert len(results) == 2
        assert results[0].url == "https://example.com/1"
        assert results[0].success is True
        assert results[0].cache_path == Path("/cache/1.md")
        assert results[0].error is None
        assert results[1].url == "https://example.com/2"
        assert results[1].success is True
        assert results[1].cache_path == Path("/cache/2.md")
        assert results[1].error is None

        # Verify fetch_documents called with correct args
        fetcher.fetch_documents.assert_called_once_with(
            ["https://example.com/1", "https://example.com/2"], progress=False
        )

    @pytest.mark.asyncio
    async def test_fetch_urls_markdown_failures(self):
        """Handle failed fetches with None document values."""
        fetcher = MagicMock()
        fetcher.fetch_documents = AsyncMock(
            return_value={
                "https://example.com/1": MagicMock(content_markdown="Content 1"),
                "https://example.com/2": None,  # Failed fetch
                "https://example.com/3": MagicMock(content_markdown="Content 3"),
            }
        )
        fetcher.cache.get_file_path = AsyncMock(
            side_effect=lambda url, content_type: Path(
                f"/cache/{url.split('/')[-1]}.md"
            )
        )

        results = await fetch_urls_markdown(
            fetcher,
            ["https://example.com/1", "https://example.com/2", "https://example.com/3"],
            verbose=False,
        )

        # Verify mixed success/failure results
        assert len(results) == 3
        assert results[0].success is True
        assert results[1].success is False
        assert results[1].cache_path is None
        assert results[1].error == "Failed to fetch URL"
        assert results[2].success is True

    @pytest.mark.asyncio
    async def test_fetch_urls_markdown_with_progress(self):
        """Progress flag passed to fetch_documents."""
        fetcher = MagicMock()
        fetcher.fetch_documents = AsyncMock(
            return_value={
                "https://example.com/1": MagicMock(content_markdown="Content")
            }
        )
        fetcher.cache.get_file_path = AsyncMock(return_value=Path("/cache/1.md"))

        await fetch_urls_markdown(fetcher, ["https://example.com/1"], verbose=True)

        # Verify progress=True passed to fetch_documents
        fetcher.fetch_documents.assert_called_once_with(
            ["https://example.com/1"], progress=True
        )

    @pytest.mark.asyncio
    async def test_fetch_urls_markdown_empty_list(self):
        """Handle empty URL list."""
        fetcher = MagicMock()
        fetcher.fetch_documents = AsyncMock(return_value={})

        results = await fetch_urls_markdown(fetcher, [], verbose=False)

        assert results == []


class TestFetchUrlsChunks:
    """Tests for fetch_urls_chunks function with mocked PageFetcher."""

    @pytest.mark.asyncio
    async def test_fetch_urls_chunks_success(self):
        """Successfully fetch and chunk multiple URLs."""
        fetcher = MagicMock()
        # Mock get_chunks_with_embeddings to return chunks
        fetcher.get_chunks_with_embeddings = AsyncMock(
            return_value=[
                MagicMock(text="Chunk 1", embedding=[0.1, 0.2]),
                MagicMock(text="Chunk 2", embedding=[0.3, 0.4]),
            ]
        )
        fetcher.cache.get_file_path = AsyncMock(
            side_effect=lambda url, content_type: Path(
                f"/cache/{url.split('/')[-1]}.chunks"
            )
        )

        results = await fetch_urls_chunks(
            fetcher, ["https://example.com/1", "https://example.com/2"], verbose=False
        )

        # Verify results
        assert len(results) == 2
        assert results[0].url == "https://example.com/1"
        assert results[0].success is True
        assert results[0].cache_path == Path("/cache/1.chunks")
        assert results[0].error is None
        assert results[1].url == "https://example.com/2"
        assert results[1].success is True

        # Verify get_chunks_with_embeddings called for each URL
        assert fetcher.get_chunks_with_embeddings.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_urls_chunks_with_failures(self):
        """Handle chunking failures with exceptions."""
        fetcher = MagicMock()
        # First URL succeeds, second fails
        fetcher.get_chunks_with_embeddings = AsyncMock(
            side_effect=[
                [MagicMock(text="Chunk", embedding=[0.1])],  # Success
                Exception("Chunking failed"),  # Failure
            ]
        )
        fetcher.cache.get_file_path = AsyncMock(return_value=Path("/cache/1.chunks"))

        results = await fetch_urls_chunks(
            fetcher, ["https://example.com/1", "https://example.com/2"], verbose=False
        )

        # Verify mixed results
        assert len(results) == 2
        assert results[0].success is True
        assert results[0].cache_path == Path("/cache/1.chunks")
        assert results[1].success is False
        assert results[1].cache_path is None
        assert "Failed to chunk URL" in results[1].error

    @pytest.mark.asyncio
    async def test_fetch_urls_chunks_with_progress(self):
        """Progress flag passed to get_chunks_with_embeddings."""
        fetcher = MagicMock()
        fetcher.get_chunks_with_embeddings = AsyncMock(
            return_value=[MagicMock(text="Chunk")]
        )
        fetcher.cache.get_file_path = AsyncMock(return_value=Path("/cache/1.chunks"))

        await fetch_urls_chunks(fetcher, ["https://example.com/1"], verbose=True)

        # Verify progress=True passed
        fetcher.get_chunks_with_embeddings.assert_called_once_with(
            "https://example.com/1", progress=True, fail_fast=False
        )

    @pytest.mark.asyncio
    async def test_fetch_urls_chunks_empty_list(self):
        """Handle empty URL list."""
        fetcher = MagicMock()

        results = await fetch_urls_chunks(fetcher, [], verbose=False)

        assert results == []
        fetcher.get_chunks_with_embeddings.assert_not_called()


class TestRunFetch:
    """Tests for run_fetch orchestrator function."""

    @pytest.mark.asyncio
    async def test_run_fetch_markdown_mode(self, tmp_path):
        """Orchestrate markdown fetch via run_fetch."""
        # Create minimal config
        config = MagicMock()
        config.abspath.return_value = tmp_path / "cache"

        with patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class:
            # Setup mock fetcher instance
            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher
            mock_fetcher.fetch_documents = AsyncMock(
                return_value={
                    "https://example.com/1": MagicMock(content_markdown="Content")
                }
            )
            mock_fetcher.cache.get_file_path = AsyncMock(
                return_value=Path("/cache/1.md")
            )

            # Run fetch in markdown mode
            results = await run_fetch(
                config, ["https://example.com/1"], chunk=False, verbose=False
            )

            # Verify PageFetcher created with config
            mock_fetcher_class.assert_called_once_with(
                config, show_status=False, verbose=False
            )

            # Verify results
            assert len(results) == 1
            assert results[0].success is True
            assert results[0].cache_path == Path("/cache/1.md")

    @pytest.mark.asyncio
    async def test_run_fetch_chunk_mode(self, tmp_path):
        """Orchestrate chunk fetch via run_fetch."""
        config = MagicMock()
        config.abspath.return_value = tmp_path / "cache"

        with patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class:
            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher
            mock_fetcher.get_chunks_with_embeddings = AsyncMock(
                return_value=[MagicMock(text="Chunk")]
            )
            mock_fetcher.cache.get_file_path = AsyncMock(
                return_value=Path("/cache/1.chunks")
            )

            # Run fetch in chunk mode
            results = await run_fetch(
                config, ["https://example.com/1"], chunk=True, verbose=False
            )

            # Verify results
            assert len(results) == 1
            assert results[0].success is True
            assert results[0].cache_path == Path("/cache/1.chunks")

    @pytest.mark.asyncio
    async def test_run_fetch_with_verbose(self, tmp_path):
        """Verbose flag propagated to PageFetcher."""
        config = MagicMock()
        config.abspath.return_value = tmp_path / "cache"

        with patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class:
            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher
            mock_fetcher.fetch_documents = AsyncMock(return_value={})

            await run_fetch(config, [], chunk=False, verbose=True)

            # Verify verbose passed to PageFetcher constructor
            mock_fetcher_class.assert_called_once_with(
                config, show_status=True, verbose=True
            )


class TestFetchWithRealCache:
    """Integration test with real PageFetcher and cache."""

    @pytest.mark.asyncio
    async def test_fetch_with_real_cache(self, tmp_path):
        """Integration test: fetch creates cache files at reported paths.

        This test uses a real config and PageFetcher, but may fail without
        network access. It validates that successful fetches create cache files
        at the paths returned in FetchResult.cache_path.
        """
        pytest.skip("Integration test requires network access and may be slow")

        # Create real config with temporary cache directory
        from interaction_finder.settings import IfetcherConfig

        config_data = {"output": {"cache": str(tmp_path / "cache")}}
        config = IfetcherConfig(**config_data)
        config._dir = tmp_path

        # Use a reliable URL for testing (example.com is always available)
        test_url = "https://example.com"

        # Run fetch with real PageFetcher
        results = await run_fetch(config, [test_url], chunk=False, verbose=False)

        # Verify results
        assert len(results) == 1
        result = results[0]

        if result.success:
            # Verify cache file exists at reported path
            assert result.cache_path is not None
            assert result.cache_path.exists()
            assert result.cache_path.is_file()
            # Verify content is non-empty markdown
            content = result.cache_path.read_text()
            assert len(content) > 0
        else:
            # Network failure is acceptable for this test
            assert result.error is not None
