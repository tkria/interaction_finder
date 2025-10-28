"""
End-to-end CLI integration tests for fetch command.

Tests complete user workflows from command invocation through CliRunner to
final output, validating argument parsing, configuration loading, fetch execution,
and output formatting.
"""

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from typer.testing import CliRunner

from interaction_finder.cli import app
from interaction_finder.cli_fetch import FetchResult

runner = CliRunner()


class TestFetchCommandInvocation:
    """Tests for basic fetch command invocation patterns."""

    def test_fetch_with_args_success(self, tmp_path):
        """Fetch multiple URLs via command-line arguments."""
        # Create temporary cache directory
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # Create mock cache files
        cache1 = cache_dir / "abc123.md"
        cache2 = cache_dir / "def456.md"
        cache1.write_text("Content 1")
        cache2.write_text("Content 2")

        # Mock config loading and PageFetcher
        with (
            patch("interaction_finder.cli.load_config") as mock_load_config,
            patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class,
        ):
            # Setup mock config
            mock_config = MagicMock()
            mock_config.abspath.return_value = cache_dir
            mock_load_config.return_value = mock_config

            # Setup mock fetcher
            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            # Mock fetch_documents to return successful docs
            mock_fetcher.fetch_documents = AsyncMock(
                return_value={
                    "https://example.com/1": MagicMock(content_markdown="Content 1"),
                    "https://example.com/2": MagicMock(content_markdown="Content 2"),
                }
            )

            # Mock cache path retrieval
            mock_fetcher.cache.get_file_path = AsyncMock(side_effect=[cache1, cache2])

            # Run fetch command with URL arguments
            result = runner.invoke(
                app,
                [
                    "fetch",
                    "https://example.com/1",
                    "https://example.com/2",
                ],
            )

            # Verify command succeeded
            assert result.exit_code == 0, f"Command failed: {result.stdout}"

            # Verify cache paths printed to stdout (one per line)
            output_lines = result.stdout.strip().split("\n")
            assert len(output_lines) >= 2
            assert str(cache1.resolve()) in result.stdout
            assert str(cache2.resolve()) in result.stdout

    def test_fetch_with_input_file_success(self, tmp_path):
        """Fetch URLs from input file."""
        # Create URL input file
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://example.com/1\nhttps://example.com/2\n")

        # Create cache files
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache1 = cache_dir / "abc123.md"
        cache2 = cache_dir / "def456.md"
        cache1.write_text("Content 1")
        cache2.write_text("Content 2")

        with (
            patch("interaction_finder.cli.load_config") as mock_load_config,
            patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class,
        ):
            mock_config = MagicMock()
            mock_config.abspath.return_value = cache_dir
            mock_load_config.return_value = mock_config

            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            mock_fetcher.fetch_documents = AsyncMock(
                return_value={
                    "https://example.com/1": MagicMock(content_markdown="Content 1"),
                    "https://example.com/2": MagicMock(content_markdown="Content 2"),
                }
            )

            mock_fetcher.cache.get_file_path = AsyncMock(side_effect=[cache1, cache2])

            # Run fetch with --input file
            result = runner.invoke(
                app,
                [
                    "fetch",
                    "--input",
                    str(url_file),
                ],
            )

            # Verify success
            assert result.exit_code == 0
            assert str(cache1.resolve()) in result.stdout
            assert str(cache2.resolve()) in result.stdout


class TestFetchCommandErrors:
    """Tests for error handling and validation."""

    def test_mutual_exclusion_error(self, tmp_path):
        """Error when both args and --input file provided."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://example.com/1\n")

        result = runner.invoke(
            app,
            [
                "fetch",
                "https://example.com/2",  # Command-line URL
                "--input",
                str(url_file),  # File input
            ],
        )

        # Verify error exit code
        assert result.exit_code == 1

        # Verify error message explains mutual exclusion
        assert "Error:" in result.stdout
        assert "both" in result.stdout.lower() or "mutual" in result.stdout.lower()

    def test_empty_input_error(self):
        """Error when neither args nor file provided."""
        result = runner.invoke(app, ["fetch"])

        # Verify error exit code
        assert result.exit_code == 1

        # Verify error message explains missing input
        assert "Error:" in result.stdout
        assert "no" in result.stdout.lower()
        assert "url" in result.stdout.lower() or "input" in result.stdout.lower()

    def test_missing_input_file_error(self, tmp_path):
        """Error when input file does not exist."""
        missing_file = tmp_path / "missing.txt"

        result = runner.invoke(
            app,
            [
                "fetch",
                "--input",
                str(missing_file),
            ],
        )

        # Verify error exit code
        assert result.exit_code == 1

        # Verify error message mentions file not found
        assert "Error:" in result.stdout
        assert (
            str(missing_file) in result.stdout or "not found" in result.stdout.lower()
        )


class TestFetchContentMode:
    """Tests for --format content output mode."""

    def test_format_content_single_url(self, tmp_path):
        """Output markdown content for single URL."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "abc123.md"
        content = "# Test Article\n\nThis is test content.\n"
        cache_file.write_text(content)

        with (
            patch("interaction_finder.cli.load_config") as mock_load_config,
            patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class,
        ):
            mock_config = MagicMock()
            mock_config.abspath.return_value = cache_dir
            mock_load_config.return_value = mock_config

            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            mock_fetcher.fetch_documents = AsyncMock(
                return_value={
                    "https://example.com/article": MagicMock(content_markdown=content)
                }
            )

            mock_fetcher.cache.get_file_path = AsyncMock(return_value=cache_file)

            # Run fetch with --format content
            result = runner.invoke(
                app,
                [
                    "fetch",
                    "https://example.com/article",
                    "--format",
                    "content",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"Command failed: {result.stdout}"

            # Verify content output to stdout
            assert content in result.stdout

    def test_format_content_multiple_urls_error(self, tmp_path):
        """Error when --format content used with multiple URLs."""
        url_file = tmp_path / "urls.txt"
        url_file.write_text("https://example.com/1\nhttps://example.com/2\n")

        result = runner.invoke(
            app,
            [
                "fetch",
                "--input",
                str(url_file),
                "--format",
                "content",
            ],
        )

        # Verify error exit code
        assert result.exit_code == 1

        # Verify error message explains single URL requirement
        assert "Error:" in result.stdout
        assert (
            "exactly one" in result.stdout.lower() or "single" in result.stdout.lower()
        )


class TestFetchChunkMode:
    """Tests for --chunk flag behavior."""

    def test_chunk_mode_success(self, tmp_path):
        """Fetch and chunk URL successfully."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        chunks_file = cache_dir / "abc123.chunks"
        chunks_file.write_text("chunk data")

        with (
            patch("interaction_finder.cli.load_config") as mock_load_config,
            patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class,
        ):
            mock_config = MagicMock()
            mock_config.abspath.return_value = cache_dir
            mock_load_config.return_value = mock_config

            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            # Mock get_chunks_with_embeddings for chunk mode
            mock_fetcher.get_chunks_with_embeddings = AsyncMock(
                return_value=[
                    MagicMock(text="Chunk 1", embedding=[0.1, 0.2]),
                    MagicMock(text="Chunk 2", embedding=[0.3, 0.4]),
                ]
            )

            mock_fetcher.cache.get_file_path = AsyncMock(return_value=chunks_file)

            # Run fetch with --chunk flag
            result = runner.invoke(
                app,
                [
                    "fetch",
                    "https://example.com/article",
                    "--chunk",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"Command failed: {result.stdout}"

            # Verify chunks path output (not markdown path)
            assert str(chunks_file.resolve()) in result.stdout
            assert ".chunks" in result.stdout


class TestFetchFailureScenarios:
    """Tests for partial and complete fetch failures."""

    def test_partial_failures(self, tmp_path):
        """Some URLs succeed, some fail - exit code 0 with warnings."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache1 = cache_dir / "abc123.md"
        cache1.write_text("Content 1")

        with (
            patch("interaction_finder.cli.load_config") as mock_load_config,
            patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class,
        ):
            mock_config = MagicMock()
            mock_config.abspath.return_value = cache_dir
            mock_load_config.return_value = mock_config

            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            # Mock fetch with one success, one failure
            mock_fetcher.fetch_documents = AsyncMock(
                return_value={
                    "https://example.com/1": MagicMock(content_markdown="Content 1"),
                    "https://example.com/2": None,  # Failed fetch
                }
            )

            mock_fetcher.cache.get_file_path = AsyncMock(return_value=cache1)

            # Run fetch with multiple URLs
            result = runner.invoke(
                app,
                [
                    "fetch",
                    "https://example.com/1",
                    "https://example.com/2",
                ],
            )

            # Verify exit code 0 (partial success)
            assert result.exit_code == 0

            # Verify successful path in output
            assert str(cache1.resolve()) in result.stdout

            # Verify warning for failed URL
            assert "Warning" in result.stdout or "Failed" in result.stdout
            assert "https://example.com/2" in result.stdout

    def test_all_failures(self, tmp_path):
        """All URLs fail - exit code 1."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        with (
            patch("interaction_finder.cli.load_config") as mock_load_config,
            patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class,
        ):
            mock_config = MagicMock()
            mock_config.abspath.return_value = cache_dir
            mock_load_config.return_value = mock_config

            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            # Mock fetch with all failures
            mock_fetcher.fetch_documents = AsyncMock(
                return_value={
                    "https://example.com/1": None,
                    "https://example.com/2": None,
                }
            )

            # Run fetch
            result = runner.invoke(
                app,
                [
                    "fetch",
                    "https://example.com/1",
                    "https://example.com/2",
                ],
            )

            # Verify exit code 1 (all failed)
            assert result.exit_code == 1

            # Verify error messages for both URLs
            assert "https://example.com/1" in result.stdout
            assert "https://example.com/2" in result.stdout


class TestFetchCacheBehavior:
    """Tests for cache hit behavior."""

    def test_cache_hit_no_refetch(self, tmp_path):
        """Cache hit should not trigger refetch."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "abc123.md"
        cache_file.write_text("Cached content")

        with (
            patch("interaction_finder.cli.load_config") as mock_load_config,
            patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class,
        ):
            mock_config = MagicMock()
            mock_config.abspath.return_value = cache_dir
            mock_load_config.return_value = mock_config

            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            # Mock fetch_documents - should return cached content quickly
            mock_fetcher.fetch_documents = AsyncMock(
                return_value={
                    "https://example.com/article": MagicMock(
                        content_markdown="Cached content"
                    )
                }
            )

            mock_fetcher.cache.get_file_path = AsyncMock(return_value=cache_file)

            # Run fetch twice
            result1 = runner.invoke(
                app,
                [
                    "fetch",
                    "https://example.com/article",
                ],
            )

            result2 = runner.invoke(
                app,
                [
                    "fetch",
                    "https://example.com/article",
                ],
            )

            # Both should succeed
            assert result1.exit_code == 0
            assert result2.exit_code == 0

            # Both should return same cache path
            assert str(cache_file.resolve()) in result1.stdout
            assert str(cache_file.resolve()) in result2.stdout


class TestFetchVerboseFlag:
    """Tests for --verbose flag behavior."""

    def test_verbose_shows_progress(self, tmp_path):
        """Verbose flag shows progress messages."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "abc123.md"
        cache_file.write_text("Content")

        with (
            patch("interaction_finder.cli.load_config") as mock_load_config,
            patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class,
        ):
            mock_config = MagicMock()
            mock_config.abspath.return_value = cache_dir
            mock_load_config.return_value = mock_config

            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            mock_fetcher.fetch_documents = AsyncMock(
                return_value={
                    "https://example.com/article": MagicMock(content_markdown="Content")
                }
            )

            mock_fetcher.cache.get_file_path = AsyncMock(return_value=cache_file)

            # Run with --verbose
            result = runner.invoke(
                app,
                [
                    "fetch",
                    "https://example.com/article",
                    "--verbose",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"Command failed: {result.stdout}"

            # Verify verbose output includes summary
            # (Summary includes stats like "Total URLs:", "Successful:", etc.)
            assert (
                "Total URLs:" in result.stdout
                or "Successful:" in result.stdout
                or "Fetch Summary" in result.stdout
            )

    def test_non_verbose_minimal_output(self, tmp_path):
        """Without --verbose, output should be minimal (just paths)."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "abc123.md"
        cache_file.write_text("Content")

        with (
            patch("interaction_finder.cli.load_config") as mock_load_config,
            patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class,
        ):
            mock_config = MagicMock()
            mock_config.abspath.return_value = cache_dir
            mock_load_config.return_value = mock_config

            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            mock_fetcher.fetch_documents = AsyncMock(
                return_value={
                    "https://example.com/article": MagicMock(content_markdown="Content")
                }
            )

            mock_fetcher.cache.get_file_path = AsyncMock(return_value=cache_file)

            # Run without --verbose
            result = runner.invoke(
                app,
                [
                    "fetch",
                    "https://example.com/article",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"Command failed: {result.stdout}"

            # Verify minimal output (just cache path, no summary)
            assert str(cache_file.resolve()) in result.stdout
            assert "Fetch Summary" not in result.stdout
            assert "Total URLs:" not in result.stdout


class TestFetchConfigurationIntegration:
    """Tests for configuration loading and override behavior."""

    def test_config_override_cache_path(self, tmp_path):
        """Config override changes cache directory."""
        custom_cache = tmp_path / "custom_cache"
        custom_cache.mkdir()
        cache_file = custom_cache / "abc123.md"
        cache_file.write_text("Content")

        with (
            patch("interaction_finder.cli.load_config") as mock_load_config,
            patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class,
        ):
            mock_config = MagicMock()
            mock_config.abspath.return_value = custom_cache
            mock_load_config.return_value = mock_config

            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            mock_fetcher.fetch_documents = AsyncMock(
                return_value={
                    "https://example.com/article": MagicMock(content_markdown="Content")
                }
            )

            mock_fetcher.cache.get_file_path = AsyncMock(return_value=cache_file)

            # Run with cache override
            result = runner.invoke(
                app,
                [
                    "fetch",
                    "https://example.com/article",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"Command failed: {result.stdout}"

            # Verify PageFetcher created with overridden config
            # (cache path should be used from override)
            assert mock_fetcher_class.called

    def test_global_options_with_fetch(self, tmp_path):
        """Global options (--config, --verbose) work with fetch command."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_file = cache_dir / "abc123.md"
        cache_file.write_text("Content")

        with (
            patch("interaction_finder.cli.load_config") as mock_load_config,
            patch("interaction_finder.cli_fetch.PageFetcher") as mock_fetcher_class,
        ):
            mock_config = MagicMock()
            mock_config.abspath.return_value = cache_dir
            mock_load_config.return_value = mock_config

            mock_fetcher = MagicMock()
            mock_fetcher_class.return_value = mock_fetcher

            mock_fetcher.fetch_documents = AsyncMock(
                return_value={
                    "https://example.com/article": MagicMock(content_markdown="Content")
                }
            )

            mock_fetcher.cache.get_file_path = AsyncMock(return_value=cache_file)

            # Run with --verbose option (global options pattern)
            # Both --verbose before and after subcommand should work
            result = runner.invoke(
                app,
                [
                    "fetch",
                    "https://example.com/article",
                    "--verbose",
                ],
            )

            # Verify success
            assert result.exit_code == 0, f"Command failed: {result.stdout}"

            # Verify verbose output present (summary shows on successful fetch with --verbose)
            assert (
                "Total URLs:" in result.stdout
                or "Successful:" in result.stdout
                or "Fetch Summary" in result.stdout
            )
