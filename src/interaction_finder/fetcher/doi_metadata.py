"""DOI metadata fetching via OpenAlex API."""

import asyncio
from typing import Optional

import httpx
import logfire


async def fetch_doi_metadata(doi: str, timeout: int = 10) -> Optional[dict]:
    """
    Fetch publication date from OpenAlex API.

    Parameters:
        doi: Digital Object Identifier
        timeout: Request timeout in seconds

    Returns:
        Dict with publication_date, or None on failure
    """
    if not doi:
        return None

    # Normalize DOI to URL format
    doi_clean = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    url = f"https://api.openalex.org/works/https://doi.org/{doi_clean}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(5):
            try:
                response = await client.get(
                    url,
                    params={
                        "select": "publication_date",
                        "mailto": "openalex@example.com",
                    },
                )

                # Retry on rate limit with exponential backoff
                if response.status_code in {403, 429} and attempt < 4:
                    await asyncio.sleep(2**attempt)
                    continue

                # Return None for expected failures
                if response.status_code != 200:
                    logfire.debug(
                        f"OpenAlex returned {response.status_code} for DOI: {doi_clean}"
                    )
                    return None

                data = response.json()
                return {
                    "doi": doi_clean,
                    "publication_date": data.get("publication_date"),
                }

            except (httpx.TimeoutException, httpx.HTTPError):
                if attempt < 4:
                    await asyncio.sleep(2**attempt)
                    continue
                logfire.debug(f"Failed to fetch DOI metadata for {doi_clean}")
                return None
            except Exception as e:
                logfire.warning(f"Unexpected error fetching DOI {doi_clean}: {e}")
                return None

    return None
