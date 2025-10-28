"""
Fetch command implementation for interaction-finder CLI.

This module provides URL collection and validation logic for the fetch subcommand,
supporting both command-line argument input and file-based input. It also provides
fetch orchestration using PageFetcher for markdown and chunk content retrieval.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from interaction_finder.fetcher import PageFetcher
from interaction_finder.settings import IfetcherConfig


def read_urls_from_file(path: Path) -> List[str]:
    """
    Read URLs from a text file, one URL per line.

    Empty lines and leading/trailing whitespace are stripped and ignored.
    Supports UTF-8 encoding for international URLs.

    Parameters:
        path: Path to the URL file

    Returns:
        List of URL strings (whitespace stripped, empty lines removed)

    Raises:
        FileNotFoundError: If the specified file does not exist
    """
    # Check file existence and raise clear error
    if not path.exists():
        raise FileNotFoundError(f"URL file not found: {path}")

    # Read file with UTF-8 encoding
    with open(path, "r", encoding="utf-8") as f:
        # Strip whitespace and filter empty lines
        urls = []
        for line in f:
            stripped = line.strip()
            if stripped:  # Skip empty lines
                urls.append(stripped)

    return urls


def collect_urls(
    args_urls: Optional[List[str]], input_file: Optional[Path]
) -> List[str]:
    """
    Collect URLs from command-line arguments or file input.

    Enforces mutual exclusion: either provide URLs via args_urls OR via input_file,
    but not both. At least one source must be provided.

    Parameters:
        args_urls: List of URLs from command-line arguments (or None/empty)
        input_file: Path to file containing URLs (or None)

    Returns:
        List of URL strings ready for fetching

    Raises:
        ValueError: If both args_urls and input_file provided (mutual exclusion)
        ValueError: If no URLs provided (neither args nor file)
        FileNotFoundError: If input_file specified but does not exist
    """
    # Normalize args_urls to empty list if None
    if args_urls is None:
        args_urls = []

    # Check mutual exclusion: both sources provided
    has_args = bool(args_urls)
    has_file = input_file is not None

    if has_args and has_file:
        raise ValueError(
            "Cannot specify both command-line URLs and --input file. "
            "Please provide URLs via one method only."
        )

    # Check empty input: neither source provided
    if not has_args and not has_file:
        raise ValueError(
            "No URLs provided. Please specify URLs via command-line arguments "
            "or use --input to provide a file."
        )

    # Collect URLs from appropriate source
    if has_file:
        # Read from file (raises FileNotFoundError if missing)
        return read_urls_from_file(input_file)
    else:
        # Return args URLs directly
        return args_urls


@dataclass
class FetchResult:
    """Result of fetching a single URL with success/failure tracking."""

    url: str
    cache_path: Optional[Path]
    success: bool
    error: Optional[str]


async def fetch_urls_markdown(
    fetcher: PageFetcher, urls: List[str], verbose: bool
) -> List[FetchResult]:
    """
    Fetch markdown content for multiple URLs with per-URL success/failure tracking.

    Uses PageFetcher.fetch_documents for batch fetching with progress display.
    Retrieves cache paths using URLCache.get_file_path for successful fetches.

    Parameters:
        fetcher: PageFetcher instance configured with cache and client
        urls: List of URLs to fetch
        verbose: Show progress bar during fetching

    Returns:
        List of FetchResult objects, one per URL in input order

    Raises:
        No exceptions raised - failures recorded in FetchResult.error field
    """
    # Fetch all URLs with progress display controlled by verbose flag
    documents = await fetcher.fetch_documents(urls, progress=verbose)

    # Build results with cache paths for successful fetches
    results = []
    for url in urls:
        doc = documents.get(url)
        if doc is not None:
            # Success: document fetched
            # Retrieve cache path using URLCache for reliable path resolution
            cache_path = await fetcher.cache.get_file_path(url, "markdown")
            results.append(
                FetchResult(
                    url=url,
                    cache_path=cache_path,
                    success=True,
                    error=None,
                )
            )
        else:
            # Failure: document is None
            results.append(
                FetchResult(
                    url=url,
                    cache_path=None,
                    success=False,
                    error="Failed to fetch URL",
                )
            )

    return results


async def fetch_urls_chunks(
    fetcher: PageFetcher, urls: List[str], verbose: bool
) -> List[FetchResult]:
    """
    Fetch and chunk content for multiple URLs with per-URL success/failure tracking.

    Uses PageFetcher.get_chunks_with_embeddings for semantic chunking with embeddings.
    Retrieves cache paths using URLCache.get_file_path for successful fetches.

    Parameters:
        fetcher: PageFetcher instance configured with cache and client
        urls: List of URLs to fetch and chunk
        verbose: Show progress bar during fetching

    Returns:
        List of FetchResult objects, one per URL in input order

    Raises:
        No exceptions raised - failures recorded in FetchResult.error field
    """
    results = []

    for url in urls:
        try:
            # Attempt to fetch and chunk the URL
            chunks = await fetcher.get_chunks_with_embeddings(
                url, progress=verbose, fail_fast=False
            )

            # Success: chunks returned
            # Retrieve cache path using URLCache
            cache_path = await fetcher.cache.get_file_path(url, "chunks")
            results.append(
                FetchResult(
                    url=url,
                    cache_path=cache_path,
                    success=True,
                    error=None,
                )
            )
        except Exception as e:
            # Failure: exception raised during chunking
            results.append(
                FetchResult(
                    url=url,
                    cache_path=None,
                    success=False,
                    error=f"Failed to chunk URL: {str(e)}",
                )
            )

    return results


async def run_fetch(
    config: IfetcherConfig, urls: List[str], chunk: bool, verbose: bool
) -> List[FetchResult]:
    """
    Orchestrate fetch operation for multiple URLs with markdown or chunk mode.

    Creates PageFetcher instance and routes to appropriate fetch function based
    on chunk flag. Handles both markdown content fetching and semantic chunking.

    Parameters:
        config: Configuration with cache directory and fetcher settings
        urls: List of URLs to fetch
        chunk: If True, perform chunking; if False, fetch markdown only
        verbose: Show progress bars and status messages

    Returns:
        List of FetchResult objects with success/failure tracking per URL

    Raises:
        No exceptions raised - all failures recorded in results
    """
    # Create PageFetcher with configuration
    fetcher = PageFetcher(config, show_status=verbose, verbose=verbose)

    # Route to appropriate fetch function based on mode
    if chunk:
        return await fetch_urls_chunks(fetcher, urls, verbose)
    else:
        return await fetch_urls_markdown(fetcher, urls, verbose)
