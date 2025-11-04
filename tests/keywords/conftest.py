"""Pytest configuration for keywords tests."""

import os

import pytest


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


def pytest_collection_modifyitems(config, items):
    """Skip tests requiring OpenAI if real API key not available."""
    # Check if we're using the dummy key
    key = os.getenv("OPENAI_API_KEY", "")
    if key and not key.startswith("sk-test-dummy"):
        return
    skip_openai = pytest.mark.skip(reason="OpenAI API key not available")
    for item in items:
        if "requires_openai" in item.keywords:
            item.add_marker(skip_openai)
