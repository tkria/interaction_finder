"""
Tests for PubMed MeSH co-occurrence query expansion.
"""

import asyncio
import xml.etree.ElementTree as ET
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

from src.interaction_finder.search.expansion.pubmed_mesh import (
    PubMedMeshExpander,
    create_pubmed_mesh_expander,
)
from src.interaction_finder.search.base import ExpandedQuery, ExpansionTerm


@pytest.fixture
def mock_esearch_response():
    """Mock ESearch XML response with PMIDs."""
    return """<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE eSearchResult PUBLIC "-//NLM//DTD esearch 20060628//EN" "https://eutils.ncbi.nlm.nih.gov/eutils/dtd/20060628/esearch.dtd">
<eSearchResult>
    <Count>3</Count>
    <IdList>
        <Id>12345678</Id>
        <Id>12345679</Id>
        <Id>12345680</Id>
    </IdList>
</eSearchResult>"""


@pytest.fixture
def mock_efetch_response():
    """Mock EFetch XML response with MeSH terms."""
    return """<?xml version="1.0" encoding="UTF-8" ?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubMedArticle, 1st January 2023//EN" "https://dtd.nlm.nih.gov/ncbi/pubmed/out/pubmed_230101.dtd">
<PubmedArticleSet>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>12345678</PMID>
            <Article>
                <ArticleTitle>Test Article 1</ArticleTitle>
            </Article>
            <MeshHeadingList>
                <MeshHeading>
                    <DescriptorName>Hypertension, Pulmonary</DescriptorName>
                </MeshHeading>
                <MeshHeading>
                    <DescriptorName>Pulmonary Artery</DescriptorName>
                </MeshHeading>
                <MeshHeading>
                    <DescriptorName>Humans</DescriptorName>
                </MeshHeading>
            </MeshHeadingList>
        </MedlineCitation>
    </PubmedArticle>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>12345679</PMID>
            <Article>
                <ArticleTitle>Test Article 2</ArticleTitle>
            </Article>
            <MeshHeadingList>
                <MeshHeading>
                    <DescriptorName>Hypertension, Pulmonary</DescriptorName>
                </MeshHeading>
                <MeshHeading>
                    <DescriptorName>Vascular Remodeling</DescriptorName>
                </MeshHeading>
                <MeshHeading>
                    <DescriptorName>Male</DescriptorName>
                </MeshHeading>
            </MeshHeadingList>
        </MedlineCitation>
    </PubmedArticle>
    <PubmedArticle>
        <MedlineCitation>
            <PMID>12345680</PMID>
            <Article>
                <ArticleTitle>Test Article 3</ArticleTitle>
            </Article>
            <MeshHeadingList>
                <MeshHeading>
                    <DescriptorName>Hypertension, Pulmonary</DescriptorName>
                </MeshHeading>
                <MeshHeading>
                    <DescriptorName>Pulmonary Embolism</DescriptorName>
                </MeshHeading>
                <MeshHeading>
                    <DescriptorName>Female</DescriptorName>
                </MeshHeading>
            </MeshHeadingList>
        </MedlineCitation>
    </PubmedArticle>
</PubmedArticleSet>"""


class TestPubMedMeshExpander:
    """Tests for PubMedMeshExpander class."""

    def test_initialization(self):
        """Test expander initialization with various configurations."""
        expander = PubMedMeshExpander(
            max_seed_results=50,
            top_terms=10,
            tree_filters=["C"],
            min_frequency=2,
            email="test@example.com",
            api_key="test_key",
        )

        assert expander.max_seed_results == 50
        assert expander.top_terms == 10
        assert expander.tree_filters == ["C"]
        assert expander.min_frequency == 2
        assert expander.email == "test@example.com"
        assert expander.api_key == "test_key"
        assert expander.rate_limit == 10.0  # With API key

    def test_initialization_defaults(self):
        """Test expander initialization with defaults."""
        expander = PubMedMeshExpander()

        assert expander.max_seed_results == 100
        assert expander.top_terms == 5
        assert expander.tree_filters == ["C", "D"]
        assert expander.min_frequency == 3
        assert expander.rate_limit == 3.0  # Without API key

    def test_expansion_method_property(self):
        """Test that expansion_method returns correct identifier."""
        expander = PubMedMeshExpander()
        assert expander.expansion_method == "pubmed_mesh"

    def test_supports_context(self):
        """Test that supported context keys are returned."""
        expander = PubMedMeshExpander()
        context_keys = expander.supports_context()

        assert "max_seed_results" in context_keys
        assert "top_terms" in context_keys
        assert "tree_filters" in context_keys
        assert "min_frequency" in context_keys

    @pytest.mark.asyncio
    async def test_esearch_parsing(self, mock_esearch_response):
        """Test ESearch response parsing."""
        expander = PubMedMeshExpander()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_esearch_response

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_response

        # Mock _get_session to return our mock session
        expander._get_session = AsyncMock(return_value=mock_session)

        pmids = await expander._esearch("test query", max_results=10)

        assert len(pmids) == 3
        assert pmids == ["12345678", "12345679", "12345680"]

    @pytest.mark.asyncio
    async def test_esearch_empty_response(self):
        """Test ESearch with no results."""
        expander = PubMedMeshExpander()

        empty_response = """<?xml version="1.0" encoding="UTF-8" ?>
<eSearchResult>
    <Count>0</Count>
    <IdList></IdList>
</eSearchResult>"""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = empty_response

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_response

        # Mock _get_session to return our mock session
        expander._get_session = AsyncMock(return_value=mock_session)

        pmids = await expander._esearch("test query", max_results=10)

        assert len(pmids) == 0

    @pytest.mark.asyncio
    async def test_efetch_mesh_terms_parsing(self, mock_efetch_response):
        """Test EFetch MeSH term extraction."""
        expander = PubMedMeshExpander()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = mock_efetch_response

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_response

        # Mock _get_session to return our mock session
        expander._get_session = AsyncMock(return_value=mock_session)

        mesh_terms = await expander._efetch_mesh_terms(
            ["12345678", "12345679", "12345680"]
        )

        # Should extract all MeSH terms from all articles
        assert "Hypertension, Pulmonary" in mesh_terms
        assert "Pulmonary Artery" in mesh_terms
        assert "Vascular Remodeling" in mesh_terms
        assert "Pulmonary Embolism" in mesh_terms
        assert "Humans" in mesh_terms
        assert "Male" in mesh_terms
        assert "Female" in mesh_terms

        # Check frequencies - "Hypertension, Pulmonary" appears 3 times
        assert mesh_terms.count("Hypertension, Pulmonary") == 3

    @pytest.mark.asyncio
    async def test_efetch_empty_response(self):
        """Test EFetch with no MeSH terms."""
        expander = PubMedMeshExpander()

        empty_response = """<?xml version="1.0" encoding="UTF-8" ?>
<PubmedArticleSet></PubmedArticleSet>"""

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = empty_response

        mock_session = AsyncMock()
        mock_session.get.return_value = mock_response

        # Mock _get_session to return our mock session
        expander._get_session = AsyncMock(return_value=mock_session)

        mesh_terms = await expander._efetch_mesh_terms(["12345678"])

        assert len(mesh_terms) == 0

    @pytest.mark.asyncio
    async def test_expand_query_integration(
        self, mock_esearch_response, mock_efetch_response
    ):
        """Test full query expansion workflow."""
        expander = PubMedMeshExpander(max_seed_results=10, top_terms=3, min_frequency=1)

        # Mock the HTTP responses
        mock_session = AsyncMock()

        async def mock_get(url, params=None):
            mock_response = MagicMock()
            mock_response.status_code = 200

            if "esearch" in url:
                mock_response.text = mock_esearch_response
            elif "efetch" in url:
                mock_response.text = mock_efetch_response

            return mock_response

        mock_session.get.side_effect = mock_get

        # Mock _get_session to return our mock session
        expander._get_session = AsyncMock(return_value=mock_session)

        expanded = await expander.expand_query("Pulmonary Hypertension")

        assert isinstance(expanded, ExpandedQuery)
        assert expanded.original_query == "Pulmonary Hypertension"
        assert expanded.expansion_method == "pubmed_mesh"

        # Should have expansion terms (after filtering demographic terms)
        assert len(expanded.expanded_terms) > 0

        # Check that demographic terms are filtered out
        term_names = [t.term for t in expanded.expanded_terms]
        assert "Humans" not in term_names
        assert "Male" not in term_names
        assert "Female" not in term_names

        # Check that relevant terms are present
        assert "Hypertension, Pulmonary" in term_names

        # Check term properties
        for term in expanded.expanded_terms:
            assert isinstance(term, ExpansionTerm)
            assert 0.0 <= term.confidence <= 1.0
            assert term.source == "pubmed_mesh"
            assert term.category == "mesh_coindex"

    @pytest.mark.asyncio
    async def test_expand_query_empty_query(self):
        """Test expansion with empty query."""
        expander = PubMedMeshExpander()

        expanded = await expander.expand_query("")

        assert expanded.original_query == ""
        assert len(expanded.expanded_terms) == 0
        assert expanded.expansion_method == "pubmed_mesh"
        assert expanded.total_confidence == 0.0

    @pytest.mark.asyncio
    async def test_expand_query_no_pmids(self):
        """Test expansion when no PMIDs are found."""
        expander = PubMedMeshExpander()

        # Mock empty ESearch response
        mock_session = AsyncMock()
        empty_response = MagicMock()
        empty_response.status_code = 200
        empty_response.text = """<?xml version="1.0" encoding="UTF-8" ?>
<eSearchResult>
    <Count>0</Count>
    <IdList></IdList>
</eSearchResult>"""
        mock_session.get.return_value = empty_response
        expander._get_session = AsyncMock(return_value=mock_session)

        expanded = await expander.expand_query("nonexistent disease xyz123")

        assert len(expanded.expanded_terms) == 0
        assert expanded.total_confidence == 0.0

    @pytest.mark.asyncio
    async def test_expand_query_min_frequency_filter(
        self, mock_esearch_response, mock_efetch_response
    ):
        """Test that min_frequency filter works correctly."""
        expander = PubMedMeshExpander(
            max_seed_results=10,
            top_terms=10,
            min_frequency=3,  # High threshold
        )

        mock_session = AsyncMock()

        async def mock_get(url, params=None):
            mock_response = MagicMock()
            mock_response.status_code = 200

            if "esearch" in url:
                mock_response.text = mock_esearch_response
            elif "efetch" in url:
                mock_response.text = mock_efetch_response

            return mock_response

        mock_session.get.side_effect = mock_get
        expander._get_session = AsyncMock(return_value=mock_session)

        expanded = await expander.expand_query("test")

        # Only "Hypertension, Pulmonary" appears 3 times, others appear once
        # After demographic filtering
        term_names = [t.term for t in expanded.expanded_terms]
        assert "Hypertension, Pulmonary" in term_names

        # Terms with frequency < 3 should not appear
        assert "Pulmonary Artery" not in term_names  # appears once
        assert "Vascular Remodeling" not in term_names  # appears once

    @pytest.mark.asyncio
    async def test_expand_query_with_context(
        self, mock_esearch_response, mock_efetch_response
    ):
        """Test expansion with context overrides."""
        expander = PubMedMeshExpander(
            max_seed_results=100, top_terms=5, min_frequency=3
        )

        mock_session = AsyncMock()

        async def mock_get(url, params=None):
            mock_response = MagicMock()
            mock_response.status_code = 200

            if "esearch" in url:
                mock_response.text = mock_esearch_response
            elif "efetch" in url:
                mock_response.text = mock_efetch_response

            return mock_response

        mock_session.get.side_effect = mock_get
        expander._get_session = AsyncMock(return_value=mock_session)

        # Override with context
        context = {"max_seed_results": 50, "top_terms": 2, "min_frequency": 1}

        expanded = await expander.expand_query("test", context=context)

        # Context should override defaults
        # top_terms=2 means max 2 expansion terms
        assert len(expanded.expanded_terms) <= 2

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test that rate limiting is enforced."""
        expander = PubMedMeshExpander()

        # First request should be immediate
        start = asyncio.get_event_loop().time()
        await expander._enforce_rate_limit()
        first_duration = asyncio.get_event_loop().time() - start
        assert first_duration < 0.1  # Should be instant

        # Second request should wait to enforce rate limit
        start = asyncio.get_event_loop().time()
        await expander._enforce_rate_limit()
        second_duration = asyncio.get_event_loop().time() - start

        # With rate_limit=3.0, minimum interval is 1/3 = 0.33 seconds
        assert second_duration >= 0.3  # Should wait at least ~0.33s

    @pytest.mark.asyncio
    async def test_demographic_filtering(
        self, mock_esearch_response, mock_efetch_response
    ):
        """Test that demographic terms are filtered correctly."""
        expander = PubMedMeshExpander(min_frequency=1, top_terms=20)

        mock_session = AsyncMock()

        async def mock_get(url, params=None):
            mock_response = MagicMock()
            mock_response.status_code = 200

            if "esearch" in url:
                mock_response.text = mock_esearch_response
            elif "efetch" in url:
                mock_response.text = mock_efetch_response

            return mock_response

        mock_session.get.side_effect = mock_get
        expander._get_session = AsyncMock(return_value=mock_session)

        expanded = await expander.expand_query("test")

        term_names = [t.term for t in expanded.expanded_terms]

        # All these should be filtered out
        excluded = ["Humans", "Male", "Female", "Animals", "Middle Aged", "Aged"]
        for term in excluded:
            assert term not in term_names, f"{term} should be filtered out"

    @pytest.mark.asyncio
    async def test_confidence_calculation(
        self, mock_esearch_response, mock_efetch_response
    ):
        """Test confidence score calculation."""
        expander = PubMedMeshExpander(min_frequency=1, top_terms=10)

        mock_session = AsyncMock()

        async def mock_get(url, params=None):
            mock_response = MagicMock()
            mock_response.status_code = 200

            if "esearch" in url:
                mock_response.text = mock_esearch_response
            elif "efetch" in url:
                mock_response.text = mock_efetch_response

            return mock_response

        mock_session.get.side_effect = mock_get
        expander._get_session = AsyncMock(return_value=mock_session)

        expanded = await expander.expand_query("test")

        # Check that confidence is normalized by number of PMIDs
        # "Hypertension, Pulmonary" appears 3 times in 3 PMIDs = 3/3 = 1.0
        hp_term = next(
            (t for t in expanded.expanded_terms if t.term == "Hypertension, Pulmonary"),
            None,
        )
        assert hp_term is not None
        assert hp_term.confidence == 1.0

        # Terms appearing once should have confidence = 1/3 = 0.33
        single_terms = [
            t
            for t in expanded.expanded_terms
            if t.term
            in ["Pulmonary Artery", "Vascular Remodeling", "Pulmonary Embolism"]
        ]
        for term in single_terms:
            assert abs(term.confidence - 0.33) < 0.05

    @pytest.mark.asyncio
    async def test_close_session(self):
        """Test session cleanup."""
        expander = PubMedMeshExpander()

        # Create a mock session and assign it
        mock_session = AsyncMock()
        mock_session.is_closed = False
        expander._session = mock_session  # Directly assign so close() finds it

        await expander.close()

        # Should call aclose on session
        mock_session.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Test async context manager usage."""
        async with PubMedMeshExpander() as expander:
            assert expander is not None
            assert isinstance(expander, PubMedMeshExpander)

        # Session should be closed after context exit
        # (This is tested implicitly by no errors during cleanup)


def test_create_pubmed_mesh_expander():
    """Test factory function."""
    expander = create_pubmed_mesh_expander(
        max_seed_results=50,
        top_terms=10,
        tree_filters=["C"],
        min_frequency=2,
        email="test@example.com",
        api_key="test_key",
    )

    assert isinstance(expander, PubMedMeshExpander)
    assert expander.max_seed_results == 50
    assert expander.top_terms == 10
    assert expander.tree_filters == ["C"]
    assert expander.min_frequency == 2
    assert expander.email == "test@example.com"
    assert expander.api_key == "test_key"
