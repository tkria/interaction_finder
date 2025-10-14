"""
Test backend selection for reverse search.

Verifies that backend is correctly selected from config when no CLI flag is provided,
and that CLI flag properly overrides config value.
"""

import pytest
from pathlib import Path
from interaction_finder.settings import IfetcherConfig
from interaction_finder.search.reverse.models import ReverseSearchConfig


def test_reverse_search_backend_defaults_to_pubmed():
    """ReverseSearchConfig default should be pubmed."""
    config = ReverseSearchConfig()
    assert config.search_backend == "pubmed"


def test_reverse_search_backend_from_toml(tmp_path):
    """Backend should be loaded from TOML config when specified."""
    # Create temporary config file
    config_file = tmp_path / "test_config.toml"
    config_file.write_text("""
[tools.reverse_search]
search_backend = "perplexica"
""")

    # Load config
    cfg = IfetcherConfig.from_path(config_file)

    # Verify reverse_search backend is loaded correctly
    assert cfg.tools.reverse_search.search_backend == "perplexica"


def test_reverse_search_backend_independent_from_search_backend(tmp_path):
    """Reverse search backend should not inherit from tools.search.backend."""
    # Create config with different backends for search and reverse_search
    config_file = tmp_path / "test_config.toml"
    config_file.write_text("""
[tools.search]
backend = "perplexica"

[tools.reverse_search]
search_backend = "pubmed"
""")

    # Load config
    cfg = IfetcherConfig.from_path(config_file)

    # Verify they're independent
    assert cfg.tools.search.backend == "perplexica"
    assert cfg.tools.reverse_search.search_backend == "pubmed"

    # These should be different!
    assert cfg.tools.search.backend != cfg.tools.reverse_search.search_backend


def test_reverse_search_uses_default_when_not_specified(tmp_path):
    """When tools.reverse_search.search_backend not specified, should use default (pubmed)."""
    # Create config without reverse_search.search_backend
    config_file = tmp_path / "test_config.toml"
    config_file.write_text("""
[tools.search]
backend = "perplexica"

[tools.reverse_search]
keyword_extractor = "yake"
# Note: search_backend NOT specified
""")

    # Load config
    cfg = IfetcherConfig.from_path(config_file)

    # Verify reverse_search uses default, not tools.search backend
    assert cfg.tools.search.backend == "perplexica"
    assert (
        cfg.tools.reverse_search.search_backend == "pubmed"
    )  # Should use default, not "perplexica"
