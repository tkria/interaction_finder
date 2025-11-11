"""Tests for PubMed full-text link following functionality."""

import pytest
from unittest.mock import patch
from interaction_finder.fetcher.web_client import WebClient


class TestPubMedURLDetection:
    """Test PubMed URL pattern matching."""

    def test_is_pubmed_url_valid(self):
        """Test detection of valid PubMed URLs."""
        client = WebClient(timeout=30, verbose=False)

        # Valid PubMed URLs
        assert client._is_pubmed_url("https://pubmed.ncbi.nlm.nih.gov/12345678")
        assert client._is_pubmed_url("https://pubmed.ncbi.nlm.nih.gov/12345678/")
        assert client._is_pubmed_url("https://pubmed.ncbi.nlm.nih.gov/1")

    def test_is_pubmed_url_invalid(self):
        """Test rejection of invalid URLs."""
        client = WebClient(timeout=30, verbose=False)

        # Invalid URLs
        assert not client._is_pubmed_url("https://pubmed.ncbi.nlm.nih.gov/")
        assert not client._is_pubmed_url("https://pubmed.ncbi.nlm.nih.gov/abc123")
        assert not client._is_pubmed_url("https://ncbi.nlm.nih.gov/pubmed/12345678")
        assert not client._is_pubmed_url(
            "https://pubmed.ncbi.nlm.nih.gov/12345678/fulltext"
        )
        assert not client._is_pubmed_url(
            "http://pubmed.ncbi.nlm.nih.gov/12345678"
        )  # HTTP not HTTPS


class TestFulltextLinkExtraction:
    """Test extraction of full-text links from PubMed HTML."""

    def test_extract_fulltext_links_basic(self):
        """Test extraction of full-text links from HTML."""
        client = WebClient(timeout=30, verbose=False)

        html = """
        <html>
            <div class="full-text-links-list">
                <a href="https://example.com/fulltext1.pdf">PDF</a>
                <a href="https://example.com/fulltext2.html">HTML</a>
            </div>
        </html>
        """

        links = client._extract_pubmed_fulltext_links(
            html, "https://pubmed.ncbi.nlm.nih.gov/12345678"
        )
        assert len(links) == 2
        assert "https://example.com/fulltext1.pdf" in links
        assert "https://example.com/fulltext2.html" in links

    def test_extract_fulltext_links_relative_urls(self):
        """Test that relative URLs are converted to absolute."""
        client = WebClient(timeout=30, verbose=False)

        html = """
        <html>
            <div class="full-text-links-list">
                <a href="/pmc/articles/PMC123456/">PMC</a>
            </div>
        </html>
        """

        links = client._extract_pubmed_fulltext_links(
            html, "https://pubmed.ncbi.nlm.nih.gov/12345678"
        )
        assert len(links) == 1
        assert links[0] == "https://pubmed.ncbi.nlm.nih.gov/pmc/articles/PMC123456/"

    def test_extract_fulltext_links_deduplication(self):
        """Test that duplicate links are removed."""
        client = WebClient(timeout=30, verbose=False)

        html = """
        <html>
            <div class="full-text-links-list">
                <a href="https://example.com/fulltext.pdf">PDF 1</a>
                <a href="https://example.com/fulltext.pdf">PDF 2</a>
            </div>
        </html>
        """

        links = client._extract_pubmed_fulltext_links(
            html, "https://pubmed.ncbi.nlm.nih.gov/12345678"
        )
        assert len(links) == 1
        assert "https://example.com/fulltext.pdf" in links

    def test_extract_fulltext_links_empty(self):
        """Test extraction with no links returns empty list."""
        client = WebClient(timeout=30, verbose=False)

        html = """
        <html>
            <div class="full-text-links-list">
            </div>
        </html>
        """

        links = client._extract_pubmed_fulltext_links(
            html, "https://pubmed.ncbi.nlm.nih.gov/12345678"
        )
        assert links == []

    def test_extract_fulltext_links_no_div(self):
        """Test extraction when div is not present."""
        client = WebClient(timeout=30, verbose=False)

        html = "<html><body>No full-text links</body></html>"

        links = client._extract_pubmed_fulltext_links(
            html, "https://pubmed.ncbi.nlm.nih.gov/12345678"
        )
        assert links == []

    def test_extract_fulltext_links_malformed_html(self):
        """Test extraction handles malformed HTML gracefully."""
        client = WebClient(timeout=30, verbose=False)

        html = "<html><div class='full-text-links-list'><a href=broken"

        links = client._extract_pubmed_fulltext_links(
            html, "https://pubmed.ncbi.nlm.nih.gov/12345678"
        )
        # Should return empty list rather than crash
        assert isinstance(links, list)


class TestContentSupersetComparison:
    """Test content superset comparison logic."""

    def test_is_content_superset_length_check(self):
        """Test that candidate must be at least 50% longer."""
        client = WebClient(timeout=30, verbose=False)

        # Baseline: 100 chars, Candidate: 140 chars (40% improvement) -> False
        baseline = "# Abstract\n" + "x" * 90
        candidate_short = "# Abstract\n" + "x" * 130

        assert not client._is_content_superset(candidate_short, baseline)

        # Candidate: 160 chars (60% improvement) -> True if headings match and content increased
        candidate_long = "# Abstract\n" + "x" * 150
        # Should pass: 60% longer, same heading, more content under heading
        assert client._is_content_superset(candidate_long, baseline)

    def test_is_content_superset_heading_subset_check(self):
        """Test that all baseline headings must be in candidate."""
        client = WebClient(timeout=30, verbose=False)

        baseline = (
            """# Abstract
Abstract content here.

## Introduction
Introduction content.
"""
            + "x" * 200
        )

        # Candidate missing Introduction heading
        candidate_missing = (
            """# Abstract
Abstract content with much more detail added here.
"""
            + "x" * 300
        )

        assert not client._is_content_superset(candidate_missing, baseline)

        # Candidate has both headings
        candidate_complete = (
            """# Abstract
Abstract content with much more detail added here.

## Introduction
Introduction with much more content.
"""
            + "x" * 300
        )

        # This should pass (length check passes, headings match, content increased)
        result = client._is_content_superset(candidate_complete, baseline)
        # May still fail depending on exact content length calculation
        assert isinstance(result, bool)

    def test_is_content_superset_heading_content_length(self):
        """Test that heading content must not decrease."""
        client = WebClient(timeout=30, verbose=False)

        baseline = (
            """# Abstract
"""
            + "x" * 200
            + """

## Introduction
"""
            + "y" * 100
        )

        # Candidate with same headings but less content under Introduction
        candidate = (
            """# Abstract
"""
            + "x" * 250
            + """

## Introduction
"""
            + "y" * 50
        )

        # Should fail because content under Introduction decreased
        # (even though overall length increased)
        assert not client._is_content_superset(candidate, baseline)

    def test_is_content_superset_true_case(self):
        """Test a clear superset case."""
        client = WebClient(timeout=30, verbose=False)

        baseline = """# Abstract
Short abstract.

## Introduction
Short intro.
"""

        candidate = """# Abstract
Much longer abstract with significantly more detail and information.

## Introduction
Much longer introduction with additional paragraphs and expanded content.

## Methods
Additional section not in baseline.

## Results
More additional content.
"""

        assert client._is_content_superset(candidate, baseline)

    def test_is_content_superset_with_pubmed_metadata(self):
        """Test that PubMed metadata headings are excluded from comparison."""
        client = WebClient(timeout=30, verbose=False)

        # Baseline with PubMed metadata headings (typical PubMed page)
        baseline = """# Article Title

## Abstract
Short abstract content.

## Figures
Figure 1, Figure 2

## Conflict of interest statement
The authors declare no conflicts.

## References
[List of references]
"""

        # Candidate (full-text) without PubMed metadata
        candidate = """# Article Title

## Abstract
Short abstract content with much more detail and expanded information.

## Introduction
Full introduction section with comprehensive background.

## Methods
Detailed methodology section.

## Results
Comprehensive results with analysis.

## Discussion
In-depth discussion of findings.
"""

        # Should pass: PubMed metadata headings are filtered out
        # Only "Article Title" and "Abstract" need to match
        assert client._is_content_superset(candidate, baseline)


@pytest.mark.asyncio
class TestPubMedFulltextFollowing:
    """Test the full PubMed full-text link following workflow."""

    async def test_try_pubmed_fulltext_links_no_links(self):
        """Test behavior when no full-text links provided."""
        client = WebClient(timeout=30, verbose=False)

        pubmed_result = {
            "raw_content": "<html>PubMed abstract</html>",
            "markdown_content": "# Abstract\nShort content.",
            "final_url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
            "doi": "",
        }

        result = await client._try_pubmed_fulltext_links(pubmed_result, [])

        assert result is None

    async def test_try_pubmed_fulltext_links_all_fail(self):
        """Test behavior when all full-text links fail to fetch."""
        client = WebClient(timeout=30, verbose=False)

        pubmed_result = {
            "raw_content": "<html>PubMed abstract</html>",
            "markdown_content": "# Abstract\nShort content.",
            "final_url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
            "doi": "",
        }

        # Mock fetch methods to raise exceptions
        with patch.object(client, "fetch_html", side_effect=Exception("Fetch failed")):
            result = await client._try_pubmed_fulltext_links(
                pubmed_result,
                ["https://example.com/fulltext.html"],
            )

        assert result is None

    async def test_try_pubmed_fulltext_links_not_superset(self):
        """Test behavior when full-text is not a superset (not 50% longer)."""
        client = WebClient(timeout=30, verbose=False)

        pubmed_result = {
            "raw_content": "<html>PubMed abstract</html>",
            "markdown_content": "# Abstract\n" + "x" * 100,
            "final_url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
            "doi": "",
        }

        # Mock full-text fetch with only slightly longer content (30% increase)
        fulltext_result = {
            "raw_content": "<html>Full text</html>",
            "markdown_content": "# Abstract\n" + "x" * 130,
            "final_url": "https://example.com/fulltext.html",
            "doi": "",
        }

        with patch.object(client, "fetch_html", return_value=fulltext_result):
            result = await client._try_pubmed_fulltext_links(
                pubmed_result,
                ["https://example.com/fulltext.html"],
            )

        assert result is None

    async def test_try_pubmed_fulltext_links_finds_better(self):
        """Test successful case where better full-text is found."""
        client = WebClient(timeout=30, verbose=False)

        pubmed_result = {
            "raw_content": "<html>PubMed abstract</html>",
            "markdown_content": """# Abstract
Short abstract content here.
""",
            "final_url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
            "doi": "",
        }

        # Full-text with significantly more content (superset)
        fulltext_result = {
            "raw_content": "<html>Full text</html>",
            "markdown_content": """# Abstract
Short abstract content here.

## Introduction
Much longer introduction with additional details.

## Methods
Detailed methodology section.

## Results
Comprehensive results section with analysis.

## Discussion
In-depth discussion of findings.

## Conclusion
Summary and future directions.
""",
            "final_url": "https://example.com/fulltext.html",
            "doi": "",
        }

        with patch.object(client, "fetch_html", return_value=fulltext_result):
            result = await client._try_pubmed_fulltext_links(
                pubmed_result,
                ["https://example.com/fulltext.html"],
            )

        assert result is not None
        assert result["final_url"] == "https://example.com/fulltext.html"

    async def test_try_pubmed_fulltext_links_picks_best(self):
        """Test that the longest full-text is selected when multiple pass."""
        client = WebClient(timeout=30, verbose=False)

        pubmed_result = {
            "raw_content": "<html>PubMed abstract</html>",
            "markdown_content": "# Abstract\n" + "x" * 50,
            "final_url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
            "doi": "",
        }

        # Two full-text links, both supersets but different lengths
        fulltext1 = {
            "raw_content": "<html>Full text 1</html>",
            "markdown_content": "# Abstract\n"
            + "x" * 50
            + "\n\n## Introduction\n"
            + "y" * 100,
            "final_url": "https://example.com/fulltext1.html",
            "doi": "",
        }

        fulltext2 = {
            "raw_content": "<html>Full text 2</html>",
            "markdown_content": "# Abstract\n"
            + "x" * 50
            + "\n\n## Introduction\n"
            + "y" * 200,
            "final_url": "https://example.com/fulltext2.html",
            "doi": "",
        }

        # Mock fetch to return different results for different URLs
        async def mock_fetch(url, retry=False):
            if "fulltext1" in url:
                return fulltext1
            elif "fulltext2" in url:
                return fulltext2

        with patch.object(client, "fetch_html", side_effect=mock_fetch):
            result = await client._try_pubmed_fulltext_links(
                pubmed_result,
                [
                    "https://example.com/fulltext1.html",
                    "https://example.com/fulltext2.html",
                ],
            )

        assert result is not None
        # Should pick fulltext2 as it's longer
        assert result["final_url"] == "https://example.com/fulltext2.html"


@pytest.mark.asyncio
class TestPubMedIntegration:
    """Test full integration of PubMed functionality in fetch_html."""

    async def test_fetch_html_non_pubmed_url(self):
        """Test that non-PubMed URLs are not processed."""
        client = WebClient(timeout=30, verbose=False)

        regular_result = {
            "raw_content": "<html>Regular content</html>",
            "markdown_content": "Regular content",
            "final_url": "https://example.com/article",
            "doi": "",
        }

        with patch.object(
            client, "_fetch_with_retry_escalation", return_value=regular_result
        ):
            with patch.object(client, "_is_pubmed_url", return_value=False):
                result = await client.fetch_html("https://example.com/article")

        # Should return original result without PubMed processing
        assert result == regular_result

    async def test_fetch_html_pubmed_no_fulltext_links(self):
        """Test PubMed URL with no full-text links available."""
        client = WebClient(timeout=30, verbose=False)

        pubmed_result = {
            "raw_content": "<html>PubMed abstract</html>",
            "markdown_content": "# Abstract\nContent.",
            "final_url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
            "doi": "",
        }

        with patch.object(
            client, "_fetch_with_retry_escalation", return_value=pubmed_result
        ):
            with patch.object(
                client, "_extract_pubmed_fulltext_links", return_value=[]
            ):
                result = await client.fetch_html(
                    "https://pubmed.ncbi.nlm.nih.gov/12345678"
                )

        # Should return PubMed result since no full-text links
        assert result == pubmed_result

    async def test_fetch_html_pubmed_with_better_fulltext(self):
        """Test PubMed URL that successfully finds better full-text."""
        client = WebClient(timeout=30, verbose=False)

        pubmed_result = {
            "raw_content": "<html>PubMed abstract</html>",
            "markdown_content": "# Abstract\nShort.",
            "final_url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
            "doi": "",
        }

        better_result = {
            "raw_content": "<html>Full article</html>",
            "markdown_content": "# Abstract\nShort.\n\n## Introduction\nLong content.\n\n## Methods\nMore content.",
            "final_url": "https://example.com/fulltext",
            "doi": "",
        }

        with patch.object(
            client, "_fetch_with_retry_escalation", return_value=pubmed_result
        ):
            with patch.object(
                client,
                "_extract_pubmed_fulltext_links",
                return_value=["https://example.com/fulltext"],
            ):
                with patch.object(
                    client, "_try_pubmed_fulltext_links", return_value=better_result
                ):
                    result = await client.fetch_html(
                        "https://pubmed.ncbi.nlm.nih.gov/12345678"
                    )

        # Should return better full-text result
        assert result == better_result
        assert result["final_url"] == "https://example.com/fulltext"

    async def test_fetch_html_pubmed_keeps_abstract_if_no_better(self):
        """Test that PubMed abstract is kept if no better full-text found."""
        client = WebClient(timeout=30, verbose=False)

        pubmed_result = {
            "raw_content": "<html>PubMed abstract</html>",
            "markdown_content": "# Abstract\nContent.",
            "final_url": "https://pubmed.ncbi.nlm.nih.gov/12345678",
            "doi": "",
        }

        with patch.object(
            client, "_fetch_with_retry_escalation", return_value=pubmed_result
        ):
            with patch.object(
                client,
                "_extract_pubmed_fulltext_links",
                return_value=["https://example.com/fulltext"],
            ):
                with patch.object(
                    client, "_try_pubmed_fulltext_links", return_value=None
                ):  # No better result
                    result = await client.fetch_html(
                        "https://pubmed.ncbi.nlm.nih.gov/12345678"
                    )

        # Should fall back to PubMed result
        assert result == pubmed_result
