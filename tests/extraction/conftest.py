"""Pytest configuration for extraction tests with taint-based conditional execution."""

import logging

import pytest

from interaction_finder.resources import ResourcePool
from tests.conftest_helpers import skip_unless_tainted

pytest_collection_modifyitems = skip_unless_tainted("extraction")


@pytest.fixture
def resource_pool():
    """Create a fresh ResourcePool for testing."""
    return ResourcePool()


@pytest.fixture
def sample_resource_pool():
    """Create a ResourcePool with sample documents."""
    pool = ResourcePool()

    pool.add(
        url="https://example.com/paper1",
        title="BRCA1 and Breast Cancer Study",
        document_text="""
        BRCA1 is a tumor suppressor gene. Mutations in BRCA1 are strongly
        associated with increased risk of breast cancer. Multiple epidemiological
        studies have confirmed the BRCA1-breast cancer association.
        """,
    )

    pool.add(
        url="https://example.com/paper2",
        title="Review of Cancer Genetics",
        document_text="""
        The BRCA1 gene plays a critical role in DNA repair. Loss of BRCA1
        function leads to genomic instability and cancer development. BRCA1
        mutations are particularly common in hereditary breast cancer.
        """,
    )

    return pool


@pytest.fixture
def test_logger():
    """Create a test logger."""
    logger = logging.getLogger("test_extraction")
    logger.setLevel(logging.DEBUG)
    return logger


@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        "entity_model": "openai:gpt-4o-mini",
        "pair_model": "openai:gpt-4o-mini",
        "assessor_model": "openai:gpt-4o-mini",
        "judge_model": "openai:gpt-4o",
    }
