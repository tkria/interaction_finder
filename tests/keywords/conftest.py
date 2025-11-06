"""Pytest configuration for keywords tests with taint-based conditional execution."""

import os

import pytest
from tests.conftest_helpers import skip_unless_tainted


def has_openai_key() -> bool:
    """Check if OpenAI API key is available."""
    return bool(os.getenv("OPENAI_API_KEY"))


# Set dummy API key if not present to allow imports during test collection
# Tests will still be skipped appropriately, but modules can be imported
if not has_openai_key():
    os.environ["OPENAI_API_KEY"] = "sk-test-dummy-key-for-test-collection-only"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_openai: mark test as requiring OpenAI API key",
    )


# Create the taint-based hook from skip_unless_tainted
_taint_based_hook = skip_unless_tainted("keywords")


def pytest_collection_modifyitems(config, items):
    """Apply both taint-based skipping and OpenAI key checking.

    First applies taint-based conditional execution, then overlays
    OpenAI API key checks for tests marked with requires_openai.
    """
    # Apply taint-based skipping first
    _taint_based_hook(config, items)
    # Then apply OpenAI key checking on top
    # Check if we're using the dummy key
    key = os.getenv("OPENAI_API_KEY", "")
    if key and not key.startswith("sk-test-dummy"):
        return
    skip_openai = pytest.mark.skip(reason="OpenAI API key not available")
    for item in items:
        if "requires_openai" in item.keywords:
            item.add_marker(skip_openai)
