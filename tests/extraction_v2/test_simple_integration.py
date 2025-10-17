"""
Simple integration test to verify the model_request implementation works.
"""

import pytest


class TestSimpleIntegration:
    """Simple integration tests."""

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
