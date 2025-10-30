"""Pytest configuration for extraction tests with conditional execution."""

from pathlib import Path

import pytest

from tests.conftest_helpers import get_skip_message, should_run_tests_for_module


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "extraction: marks tests as part of the extraction test suite (conditionally skipped)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip extraction tests if conditions aren't met."""
    # Calculate path to extraction source module
    extraction_src = (
        Path(__file__).parent.parent.parent
        / "src"
        / "interaction_finder"
        / "extraction"
    )

    if should_run_tests_for_module(extraction_src):
        # Conditions met, run all tests normally
        return

    # Skip only tests from this directory (when pytest is run from parent directories,
    # items will contain tests from all subdirectories)
    this_dir = Path(__file__).parent
    skip_marker = pytest.mark.skip(reason=get_skip_message("extraction"))
    for item in items:
        # Check if this test item belongs to our subdirectory
        if Path(item.fspath).is_relative_to(this_dir):
            item.add_marker(skip_marker)
