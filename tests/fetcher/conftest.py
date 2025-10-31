"""Pytest configuration for fetcher tests with taint-based conditional execution."""

import pytest
import tempfile
from pathlib import Path
from tests.conftest_helpers import skip_unless_tainted

pytest_collection_modifyitems = skip_unless_tainted("fetcher")


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Provide a temporary cache directory for tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
