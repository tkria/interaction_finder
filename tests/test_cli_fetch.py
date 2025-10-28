"""Tests for cli_fetch module - URL input parsing and validation."""

from pathlib import Path
import pytest

from interaction_finder.cli_fetch import read_urls_from_file, collect_urls


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
