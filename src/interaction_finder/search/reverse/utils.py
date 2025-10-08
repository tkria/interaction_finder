"""
URL normalization utilities for consistent resource matching.

This module provides URL normalization for deduplication and matching across
different URL representations. Handles lowercase conversion, trailing slash
removal, tracking parameter cleanup, and fragment stripping.
"""

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
        # Lowercase scheme, netloc (domain), and path for case-insensitive matching
        scheme = parsed.scheme.lower() if parsed.scheme else ""
        netloc = parsed.netloc.lower() if parsed.netloc else ""
        # Remove trailing slash and lowercase path
        path = parsed.path.rstrip("/").lower() if parsed.path else ""
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
