"""Pytest configuration for search tests with taint-based conditional execution."""

from tests.conftest_helpers import skip_unless_tainted

pytest_collection_modifyitems = skip_unless_tainted("search")
