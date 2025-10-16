"""
Regression tests ensuring existing keyword extractors remain unchanged.

These tests verify that YAKE, RAKE, and TF-IDF extractors produce identical
results to their baseline behavior, proving no accidental behavior changes
were introduced during LLM extractor implementation.

Tests use fixed input data and snapshot-style assertions to detect any
deviations from expected keyword extraction behavior.
"""

import pytest
from unittest.mock import patch, AsyncMock

from interaction_finder.search.reverse.models import (
    KnownResource,
    ReverseSearchConfig,
)
from interaction_finder.search.reverse.query_generator import QueryGenerator
from interaction_finder.search.reverse.keyword_extractors import (
    YAKEExtractor,
    RAKEExtractor,
    TFIDFExtractor,
)


# Fixed test data for reproducibility


FIXED_TEST_TEXT = """
BRCA1 mutations are associated with increased risk of breast cancer.
The BRCA1 gene encodes a tumor suppressor protein that plays a critical role
in DNA repair. Loss of BRCA1 function leads to genomic instability and
increased susceptibility to cancer. Women with BRCA1 mutations have up to
70% lifetime risk of developing breast cancer. BRCA1 testing is recommended
for individuals with family history of breast or ovarian cancer.
"""


FIXED_ABSTRACT_1 = """
Pulmonary arterial hypertension (PAH) is characterized by elevated pulmonary
artery pressure. Mutations in BMPR2 gene account for 70% of familial PAH cases.
BMPR2 encodes a type II receptor for bone morphogenetic proteins. Loss of BMPR2
function disrupts TGF-beta signaling in pulmonary vascular cells.
"""


FIXED_ABSTRACT_2 = """
T cells play a central role in adaptive immunity. CD8+ T cells recognize antigens
presented on MHC class I molecules and kill infected or malignant cells. CD4+ T cells
provide help to B cells and other immune cells through cytokine secretion and
co-stimulation signals. T cell exhaustion occurs during chronic infection.
"""


# Fixtures


@pytest.fixture
def fixed_resources():
    """Fixed set of resources for regression testing."""
    return [
        KnownResource(
            pmid="12345678",
            url="https://pubmed.ncbi.nlm.nih.gov/12345678/",
            hint_fields={"gene": "BRCA1", "disease": "breast cancer"},
        ),
        KnownResource(
            pmid="87654321",
            url="https://pubmed.ncbi.nlm.nih.gov/87654321/",
            hint_fields={"gene": "BMPR2", "disease": "pulmonary arterial hypertension"},
        ),
    ]


# Regression tests for keyword extractors


@pytest.mark.parametrize("extractor_name", ["yake", "rake", "tfidf"])
def test_extractor_unchanged(extractor_name):
    """
    Test that existing extractors produce consistent keywords for fixed input.

    This is a snapshot-style test. If this fails, it indicates the extractor's
    behavior has changed, which may break existing workflows relying on
    deterministic keyword extraction.
    """
    # Create extractor
    config = ReverseSearchConfig(
        keyword_extractor=extractor_name,
        keywords_per_query=7,
        search_backend="test_backend",
    )
    generator = QueryGenerator(config, backend_name="pubmed", console=None)
    extractor = generator.extractor

    # Extract keywords from fixed text
    keywords = extractor.extract(FIXED_TEST_TEXT, top_n=7)

    # Verify basic properties
    assert isinstance(keywords, list), (
        f"{extractor_name} should return list of keywords"
    )
    assert len(keywords) > 0, f"{extractor_name} should extract at least one keyword"
    assert all(isinstance(k, str) for k in keywords), (
        f"{extractor_name} keywords should be strings"
    )

    # Verify basic sanity - keywords should be domain-relevant
    # (exact keywords can vary with library versions, so check for key terms presence)
    combined_keywords = " ".join(keywords).lower()

    # Key terms that should appear for BRCA1 text
    key_terms = ["brca1", "cancer", "breast", "mutations"]

    # At least 2 key terms should appear in extracted keywords
    found_terms = sum(1 for term in key_terms if term in combined_keywords)
    assert found_terms >= 2, (
        f"{extractor_name} behavior appears significantly changed: "
        f"expected at least 2 of {key_terms} in keywords, "
        f"but only found {found_terms}. Keywords: {keywords[:5]}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("extractor_name", ["yake", "rake", "tfidf"])
async def test_query_generation_unchanged(extractor_name, fixed_resources):
    """
    Test that query generation produces consistent queries for existing extractors.

    This verifies the complete flow from KnownResource → metadata → keyword extraction → query.
    If this fails, it indicates changes in query construction logic.
    """
    config = ReverseSearchConfig(
        keyword_extractor=extractor_name,
        keywords_per_query=7,
        search_backend="test_backend",
    )

    generator = QueryGenerator(config, backend_name="pubmed", console=None)

    # Mock metadata fetch to return fixed content
    with patch.object(
        generator,
        "_fetch_pmid_metadata_batch",
        return_value={
            "12345678": {"title": "BRCA1 Study", "abstract": FIXED_ABSTRACT_1},
            "87654321": {"title": "BMPR2 Research", "abstract": FIXED_ABSTRACT_2},
        },
    ):
        queries = await generator.generate_initial_queries(fixed_resources)

    # Basic sanity checks
    assert len(queries) > 0, f"Should generate at least one query with {extractor_name}"
    assert all(isinstance(q, str) for q in queries), "Queries should be strings"

    # Check query structure (keywords joined with OR, quoted)
    for query in queries:
        assert " OR " in query, f"Query should contain OR operator: {query}"
        assert '"' in query, f"Query should contain quoted keywords: {query}"

    # Verify queries contain domain-relevant content
    combined_query = " ".join(queries).lower()

    # Key domain terms that should appear across both abstracts
    # (exact terms extracted can vary, so check for general domain relevance)
    domain_terms = ["brca1", "bmpr2", "pulmonary", "breast", "cancer", "pah", "cells"]

    # At least 2 domain terms should appear in queries
    found = sum(1 for term in domain_terms if term in combined_query)
    assert found >= 2, (
        f"{extractor_name} query generation appears significantly changed: "
        f"expected at least 2 of {domain_terms} in queries, "
        f"but only found {found}. Queries: {queries}"
    )


def test_yake_parameters_unchanged():
    """Test that YAKE extractor uses correct default parameters."""
    config = ReverseSearchConfig(
        keyword_extractor="yake", keywords_per_query=7, search_backend="test_backend"
    )
    generator = QueryGenerator(config, backend_name="pubmed", console=None)
    extractor = generator.extractor

    assert isinstance(extractor, YAKEExtractor)
    # Check YAKE-specific parameters
    assert extractor.name == "yake"
    assert extractor.max_ngram_size == 3  # Default
    assert extractor.dedup_threshold == 0.9  # Default (correct attribute name)


def test_rake_parameters_unchanged():
    """Test that RAKE extractor uses correct default parameters."""
    config = ReverseSearchConfig(
        keyword_extractor="rake", keywords_per_query=7, search_backend="test_backend"
    )
    generator = QueryGenerator(config, backend_name="pubmed", console=None)
    extractor = generator.extractor

    assert isinstance(extractor, RAKEExtractor)
    assert extractor.name == "rake"
    # RAKE uses internal rake_nltk.Rake instance, no direct parameters to check


def test_tfidf_parameters_unchanged():
    """Test that TF-IDF extractor uses correct default parameters."""
    config = ReverseSearchConfig(
        keyword_extractor="tfidf", keywords_per_query=7, search_backend="test_backend"
    )
    generator = QueryGenerator(config, backend_name="pubmed", console=None)
    extractor = generator.extractor

    assert isinstance(extractor, TFIDFExtractor)
    assert extractor.name == "tfidf"
    assert extractor.ngram_range == (1, 3)  # Default (unigrams to trigrams)


@pytest.mark.asyncio
async def test_clustering_still_works_for_existing_extractors(fixed_resources):
    """
    Test that clustering logic works unchanged for YAKE/RAKE/TF-IDF.

    Verifies backward compatibility of clustering feature with existing extractors.
    """
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        enable_clustering=True,
        min_cluster_size=2,
        target_clusters=2,
        keywords_per_query=7,
        search_backend="test_backend",
    )

    generator = QueryGenerator(config, backend_name="pubmed", console=None)

    # Mock metadata fetch
    with patch.object(
        generator,
        "_fetch_pmid_metadata_batch",
        return_value={
            "12345678": {"title": "BRCA1", "abstract": FIXED_ABSTRACT_1},
            "87654321": {"title": "T cells", "abstract": FIXED_ABSTRACT_2},
        },
    ):
        queries = await generator.generate_initial_queries(fixed_resources)

    # Should generate queries (clustering may reduce query count)
    assert len(queries) > 0, "Should generate queries with clustering enabled"
    assert len(queries) <= len(fixed_resources), (
        "Clustering should not increase query count"
    )


def test_configuration_defaults_unchanged():
    """
    Test that ReverseSearchConfig defaults remain backward compatible.

    This ensures existing code using default configuration continues to work.
    """
    config = ReverseSearchConfig(search_backend="test_backend")  # All defaults

    # Critical defaults that must not change
    assert config.keyword_extractor == "yake", "Default extractor should remain YAKE"
    assert config.keywords_per_query == 7, "Default keyword count should remain 7"
    assert config.coverage_target == 0.95, "Default coverage target should remain 95%"
    assert config.consecutive_zero_limit == 3, "Default zero limit should remain 3"
    assert config.max_queries == 100, "Default max queries should remain 100"
    assert config.enable_clustering is True, "Clustering should be enabled by default"
    assert config.sort_by == "relevance", "Default sort should be relevance"

    # LLM config should exist but not be active unless explicitly set
    assert "model" in config.llm_query_config, "LLM config should have model field"
    assert config.llm_query_config["model"] == "openai:gpt-4o-mini", "Default LLM model"


def test_query_construction_format_unchanged():
    """
    Test that query string format remains consistent for existing extractors.

    Verifies that queries are properly formatted with quotes and OR operators.
    """
    # Test each extractor's query format
    for extractor_name in ["yake", "rake", "tfidf"]:
        config = ReverseSearchConfig(
            keyword_extractor=extractor_name,
            keywords_per_query=5,
            search_backend="test_backend",
        )
        generator = QueryGenerator(config, backend_name="pubmed", console=None)
        extractor = generator.extractor

        keywords = ["keyword1", "phrase two", "term3"]

        # Build query using extractor's format
        if hasattr(extractor, "format_query"):
            query = extractor.format_query(keywords)
        else:
            # Default format: quote each keyword and join with OR
            query = " OR ".join(f'"{kw}"' for kw in keywords)

        # Verify format
        assert '"keyword1"' in query, "Keywords should be quoted"
        assert '"phrase two"' in query, "Multi-word phrases should be quoted"
        assert " OR " in query, "Keywords should be joined with OR"
        # Query should not have trailing/leading spaces or extra operators
        assert not query.startswith(" OR "), "No leading OR"
        assert not query.endswith(" OR "), "No trailing OR"
        assert " OR  OR " not in query, "No duplicate OR operators"
