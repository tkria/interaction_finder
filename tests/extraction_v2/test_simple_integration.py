"""
Simple integration test to verify the model_request implementation works.
"""

import pytest
from typing import List

from pydantic_ai import models
from pydantic_ai.models.test import TestModel

from src.interaction_finder.resources import ResourcePool
from src.interaction_finder.extraction_graph_v2.directextract import (
    extract_entities_from_documents,
)
from src.interaction_finder.extraction_graph_v2.models import EntityListOut

# Prevent accidental real LLM calls during testing
models.ALLOW_MODEL_REQUESTS = False

pytestmark = pytest.mark.anyio


@pytest.fixture
def test_documents():
    """Create simple test documents."""
    resource_pool = ResourcePool()
    doc = resource_pool.add(
        "http://example.com/doc1", "Test Document", "BRCA1 is a tumor suppressor gene."
    )
    return [doc]


@pytest.fixture
def entity_kinds():
    """Basic entity kinds."""
    return ["gene", "disease"]


class TestSimpleIntegration:
    """Simple integration tests."""

    async def test_basic_functionality_with_testmodel(
        self, test_documents, entity_kinds
    ):
        """Test that the basic function call works with TestModel."""
        test_model = TestModel(
            # The TestModel will return a default response which should be valid JSON
            custom_output_text='{"entities": [], "entity_kinds": ["gene"], "reasoning": "test"}'
        )

        try:
            result = await extract_entities_from_documents(
                test_documents, test_model, entity_kinds
            )

            # If we get here without exception, the basic function call works
            assert isinstance(result, list)

        except Exception as e:
            # Print the error for debugging
            print(f"Error occurred: {e}")
            print(f"Error type: {type(e)}")
            raise

    def test_model_request_import(self):
        """Test that model_request can be imported properly."""
        try:
            from src.interaction_finder.extraction_graph_v2.directextract import (
                extract_entities_from_documents,
            )

            # If we can import it, the basic structure is correct
            assert extract_entities_from_documents is not None
        except ImportError as e:
            pytest.fail(f"Failed to import: {e}")
