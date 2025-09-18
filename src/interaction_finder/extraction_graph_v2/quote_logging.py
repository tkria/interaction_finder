"""
Quote error logging utilities for incremental saves.
"""

import json
import logging
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import QuoteErrorRecord

logger = logging.getLogger(__name__)


def save_quote_errors_incremental(
    quote_errors: List["QuoteErrorRecord"],
    output_dir: Path,
    append_mode: bool = True,
    saved_count: int = 0,
) -> int:
    """
    Save quote errors incrementally to preserve data during processing.

    Args:
        quote_errors: List of all quote error records
        output_dir: Directory to save the file in
        append_mode: If True, append to existing file; if False, overwrite
        saved_count: Number of errors already saved (to avoid duplicates)

    Returns:
        Updated count of saved errors
    """
    if not quote_errors:
        return saved_count

    quote_errors_file = output_dir / "quote_errors.json"

    # Determine which errors are new
    if append_mode and saved_count > 0:
        # Only save errors beyond saved_count
        new_errors = quote_errors[saved_count:]
    else:
        # Save all errors (first time or overwrite mode)
        new_errors = quote_errors

    if not new_errors:
        return saved_count

    existing_errors = []
    if append_mode and quote_errors_file.exists():
        try:
            with open(quote_errors_file, "r", encoding="utf-8") as f:
                existing_errors = json.load(f)
        except (json.JSONDecodeError, IOError):
            # If file is corrupted or unreadable, start fresh
            existing_errors = []

    # Add new errors
    new_error_data = [error.to_dict() for error in new_errors]

    if append_mode:
        all_errors = existing_errors + new_error_data
    else:
        # Overwrite mode - save all current errors
        all_errors = [error.to_dict() for error in quote_errors]

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write updated file
    with open(quote_errors_file, "w", encoding="utf-8") as f:
        json.dump(all_errors, f, indent=2, ensure_ascii=False)

    logger.debug(
        f"Incremental save: {len(new_error_data)} new quote errors added to {quote_errors_file}"
    )

    # Return updated count
    return len(quote_errors)
