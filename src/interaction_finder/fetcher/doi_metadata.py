"""Metadata fetching via OpenAlex API."""

import asyncio
from typing import Literal, Optional

import httpx
import logfire


async def fetch_work_metadata(
    identifier: str,
    id_type: Literal["doi", "pmid", "pmcid", "openalex"] = "doi",
    timeout: int = 10,
    select: Optional[list[str]] = None,
) -> Optional[dict]:
    """
    Fetch metadata from OpenAlex API using various identifier types.

    Parameters:
        identifier: Work identifier (DOI, PMID, PMCID, or OpenAlex ID)
        id_type: Type of identifier ("doi", "pmid", "pmcid", "openalex")
        timeout: Request timeout in seconds
        select: List of fields to select (defaults to ["publication_date"])

    Returns:
        Dict with requested fields, or None on failure

    Examples:
        # Fetch by DOI
        metadata = await fetch_work_metadata("10.1038/s41586-021-03819-2")

        # Fetch by PubMed ID
        metadata = await fetch_work_metadata("34234979", id_type="pmid")

        # Fetch multiple fields
        metadata = await fetch_work_metadata(
            "10.1038/s41586-021-03819-2",
            select=["publication_date", "title", "authorships", "cited_by_count"]
        )
    """
    if not identifier:
        return None

    if select is None:
        select = ["publication_date"]

    # Build OpenAlex work URL based on identifier type
    if id_type == "doi":
        # Normalize DOI to URL format
        id_clean = identifier.removeprefix("https://doi.org/").removeprefix(
            "http://doi.org/"
        )
        url = f"https://api.openalex.org/works/https://doi.org/{id_clean}"
    elif id_type == "pmid":
        # PubMed ID
        url = f"https://api.openalex.org/works/pmid:{identifier}"
    elif id_type == "pmcid":
        # PubMed Central ID
        pmc_clean = identifier.removeprefix("PMC")
        url = f"https://api.openalex.org/works/pmcid:{pmc_clean}"
    elif id_type == "openalex":
        # OpenAlex ID (format: W1234567890)
        url = f"https://api.openalex.org/works/{identifier}"
    else:
        logfire.warning(f"Unknown identifier type: {id_type}")
        return None

    async with httpx.AsyncClient(timeout=timeout) as client:
        for attempt in range(5):
            try:
                response = await client.get(
                    url,
                    params={
                        "select": ",".join(select),
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
                        f"OpenAlex returned {response.status_code} for {id_type}: {identifier}"
                    )
                    return None

                data = response.json()
                # Return all fields from API response
                return data

            except (httpx.TimeoutException, httpx.HTTPError):
                if attempt < 4:
                    await asyncio.sleep(2**attempt)
                    continue
                logfire.debug(f"Failed to fetch metadata for {id_type}: {identifier}")
                return None
            except Exception as e:
                logfire.warning(
                    f"Unexpected error fetching {id_type} {identifier}: {e}"
                )
                return None

    return None


async def fetch_doi_metadata(
    doi: str, timeout: int = 10, select: Optional[list[str]] = None
) -> Optional[dict]:
    """
    Backward-compatible wrapper for fetch_work_metadata using DOI.

    Parameters:
        doi: Digital Object Identifier
        timeout: Request timeout in seconds
        select: List of fields to select (defaults to ["publication_date"])

    Returns:
        Dict with requested fields plus doi field, or None on failure
    """
    result = await fetch_work_metadata(
        doi, id_type="doi", timeout=timeout, select=select
    )
    if result:
        # Add normalized DOI to result for backward compatibility
        doi_clean = doi.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
        result["doi"] = doi_clean
    return result
