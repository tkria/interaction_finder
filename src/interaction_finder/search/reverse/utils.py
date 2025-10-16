"""
URL normalization utilities for consistent resource matching.

This module provides URL normalization for deduplication and matching across
different URL representations. Handles lowercase conversion, trailing slash
removal, tracking parameter cleanup, and fragment stripping.

Also provides migration utilities for investigation log format changes.
"""

from typing import Any, Dict
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

# Common tracking parameters to remove for URL normalization
TRACKING_PARAMS = {
    # Google Analytics
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    # Facebook
    "fbclid",
    "fb_action_ids",
    "fb_action_types",
    "fb_source",
    "fb_ref",
    # Google Ads
    "gclid",
    "gclsrc",
    # Other common trackers
    "mc_cid",
    "mc_eid",  # Mailchimp
    "ref",
    "referrer",  # Generic referrer
    "_ga",
    "_gl",  # Google Analytics client IDs
}


def normalize_url(url: str) -> str:
    """
    Normalize URL for consistent matching.

    Performs the following transformations:
    1. Lowercase scheme and domain
    2. Remove tracking parameters (utm_*, fbclid, etc.)
    3. Strip trailing slash from path
    4. Remove fragment identifier
    5. Sort query parameters for determinism

    Parameters:
        url: str - URL to normalize

    Returns:
        str - Normalized URL

    Examples:
        >>> normalize_url("HTTPS://Example.com/Page/?utm_source=twitter")
        'https://example.com/page?'

        >>> normalize_url("https://example.com/page/")
        'https://example.com/page'

        >>> normalize_url("https://example.com/paper?id=123&utm_campaign=email")
        'https://example.com/paper?id=123'

    Note:
        Idempotent: normalize_url(normalize_url(x)) == normalize_url(x)
        Handles malformed URLs gracefully without exceptions.
    """
    url = url.strip()
    if not url:
        return ""

    try:
        parsed = urlparse(url)
        # Lowercase scheme and netloc (domain) for case-insensitive matching
        scheme = parsed.scheme.lower() if parsed.scheme else ""
        netloc = parsed.netloc.lower() if parsed.netloc else ""
        # Remove trailing slash but preserve path case
        path = parsed.path.rstrip("/") if parsed.path else ""
        # Parse and filter query parameters
        # keep_blank_values=True preserves params with empty values
        query_params = parse_qs(parsed.query, keep_blank_values=True)
        # Remove tracking parameters
        filtered_params = {
            k: v for k, v in query_params.items() if k not in TRACKING_PARAMS
        }
        # Reconstruct query string
        # parse_qs returns lists of values; take first value for each parameter
        query_dict = {k: v[0] if v else "" for k, v in filtered_params.items()}
        # Sort parameters for determinism
        query = urlencode(sorted(query_dict.items())) if query_dict else ""
        # Reconstruct URL without fragment
        normalized = urlunparse((scheme, netloc, path, "", query, ""))
        return normalized

    except Exception:
        # If parsing fails, fallback to basic normalization
        # Lowercase and strip trailing slash only
        return url.lower().rstrip("/")


def synthesize_legacy_format(
    extraction_entry: Dict[str, Any], construction_entry: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Synthesize legacy query_generation entry from two-stage format.

    Merges keyword_extraction and query_construction entries into a single
    query_generation entry for backward compatibility with analysis scripts
    that expect the old single-stage format.

    Parameters:
        extraction_entry: Dict - keyword_extraction log entry
        construction_entry: Dict - query_construction log entry

    Returns:
        Dict - Synthesized query_generation entry in legacy format

    Example:
        >>> extraction = {
        ...     "stage": "keyword_extraction",
        ...     "session_id": "abc123",
        ...     "timestamp": "2025-10-10T12:00:00Z",
        ...     "query_index": 0,
        ...     "extractor_type": "yake",
        ...     "keywords": [{"keyword": "CD8", "score": 0.85}],
        ...     "input_resources": ["PMID:12345678"],
        ... }
        >>> construction = {
        ...     "stage": "query_construction",
        ...     "session_id": "abc123",
        ...     "timestamp": "2025-10-10T12:00:01Z",
        ...     "query_index": 0,
        ...     "final_query": '"CD8" OR "T cell"',
        ...     "cumulative_coverage": 0.5,
        ... }
        >>> legacy = synthesize_legacy_format(extraction, construction)
        >>> legacy["stage"]
        'query_generation'
        >>> legacy["extractor_type"]
        'yake'
        >>> legacy["final_query"]
        '"CD8" OR "T cell"'

    Note:
        Use this function when migrating analysis scripts from the old
        single-stage query_generation format to the new two-stage format.
        The synthesized entry combines fields from both stages with the
        construction timestamp (later of the two).
    """
    # Validate input stages
    if extraction_entry.get("stage") != "keyword_extraction":
        raise ValueError(
            f"Expected keyword_extraction stage, got {extraction_entry.get('stage')}"
        )
    if construction_entry.get("stage") != "query_construction":
        raise ValueError(
            f"Expected query_construction stage, got {construction_entry.get('stage')}"
        )

    # Validate same query_index
    if extraction_entry.get("query_index") != construction_entry.get("query_index"):
        raise ValueError(
            f"Query indices do not match: {extraction_entry.get('query_index')} != {construction_entry.get('query_index')}"
        )

    # Synthesize legacy entry
    legacy_entry = {
        # Use construction timestamp (later of the two stages)
        "timestamp": construction_entry["timestamp"],
        "stage": "query_generation",
        "session_id": construction_entry["session_id"],
        "query_index": construction_entry["query_index"],
        # Set query_type to "initial" (legacy default)
        "query_type": "initial",
        # Extraction fields
        "extractor_type": extraction_entry["extractor_type"],
        "keywords": extraction_entry["keywords"],
        "input_resources": extraction_entry["input_resources"],
        # Construction fields
        "final_query": construction_entry["final_query"],
        "cumulative_coverage": construction_entry["cumulative_coverage"],
        # Optional fields (may not be present in all entries)
        "cluster_id": None,  # Not tracked in new format
    }

    return legacy_entry
