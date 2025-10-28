"""
Fetch command implementation for interaction-finder CLI.

This module provides URL collection and validation logic for the fetch subcommand,
supporting both command-line argument input and file-based input.
"""

from pathlib import Path
from typing import List, Optional


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
