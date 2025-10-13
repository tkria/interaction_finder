"""
Comprehensive tests for configuration migration logic.

Tests cover all acceptance criteria:
1. New query_constructor field with default "direct"
2. New query_construction_config field with sensible defaults
3. Legacy keyword_extractor="llm" migration to extractor="none" + constructor="llm"
4. Default query_constructor="direct" inference for statistical extractors
5. Invalid combination rejection: extractor="none" + constructor="direct"
6. query_constructor="llm" requires non-empty llm_query_config
7. Clear error messages with remediation suggestions
8. Config serialization round-trip
"""

import pytest
import warnings
from pydantic import ValidationError

from interaction_finder.search.reverse import (
    ReverseSearchConfig,
    ConfigurationError,
)


# ==============================================================================
# Tests for new fields and defaults
# ==============================================================================


def test_config_has_query_constructor_field():
    """Test that ReverseSearchConfig has query_constructor field."""
    config = ReverseSearchConfig()
    assert hasattr(config, "query_constructor")
    assert config.query_constructor in ["direct", "llm"]


def test_config_query_constructor_default():
    """Test that query_constructor defaults to 'direct'."""
    config = ReverseSearchConfig()
    assert config.query_constructor == "direct"


def test_config_has_query_construction_config_field():
    """Test that ReverseSearchConfig has query_construction_config field."""
    config = ReverseSearchConfig()
    assert hasattr(config, "query_construction_config")
    assert isinstance(config.query_construction_config, dict)


def test_config_query_construction_config_defaults():
    """Test that query_construction_config has sensible defaults."""
    config = ReverseSearchConfig()
    assert config.query_construction_config["enable_fallback"] is True
    assert config.query_construction_config["include_scores_in_prompt"] is True
    assert config.query_construction_config["max_keywords_for_llm"] == 10


def test_config_query_constructor_accepts_valid_values():
    """Test that query_constructor accepts valid literal values."""
    config_direct = ReverseSearchConfig(query_constructor="direct")
    assert config_direct.query_constructor == "direct"

    config_llm = ReverseSearchConfig(
        query_constructor="llm",
        keyword_extractor="none",
    )
    assert config_llm.query_constructor == "llm"


def test_config_query_constructor_rejects_invalid_values():
    """Test that query_constructor rejects invalid literal values."""
    with pytest.raises(ValidationError) as exc_info:
        ReverseSearchConfig(query_constructor="invalid")
    assert "query_constructor" in str(exc_info.value).lower()


def test_config_keyword_extractor_includes_none():
    """Test that keyword_extractor accepts 'none' value."""
    config = ReverseSearchConfig(
        keyword_extractor="none",
        query_constructor="llm",
    )
    assert config.keyword_extractor == "none"


# ==============================================================================
# Tests for legacy migration: keyword_extractor="llm"
# ==============================================================================


def test_legacy_keyword_extractor_llm_migration():
    """Test that keyword_extractor='llm' is migrated to none + llm constructor."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = ReverseSearchConfig(keyword_extractor="llm")
        # Check migration occurred
        assert config.keyword_extractor == "none"
        assert config.query_constructor == "llm"
        # Check deprecation warning was issued
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
        assert "deprecated" in str(w[0].message).lower()
        assert "keyword_extractor='llm'" in str(w[0].message)


def test_legacy_keyword_extractor_llm_with_custom_llm_config():
    """Test that legacy llm extractor preserves custom llm_query_config."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        custom_config = {
            "model": "anthropic:claude-3-opus",
            "temperature": 0.3,
        }
        config = ReverseSearchConfig(
            keyword_extractor="llm",
            llm_query_config=custom_config,
        )
        assert config.keyword_extractor == "none"
        assert config.query_constructor == "llm"
        assert config.llm_query_config == custom_config


# ==============================================================================
# Tests for backward compatibility: statistical extractors
# ==============================================================================


def test_backward_compatible_yake_extractor():
    """Test that legacy yake config works unchanged (infers direct constructor)."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        keywords_per_query=5,
    )
    assert config.keyword_extractor == "yake"
    assert config.query_constructor == "direct"  # Inferred default


def test_backward_compatible_rake_extractor():
    """Test that legacy rake config works unchanged (infers direct constructor)."""
    config = ReverseSearchConfig(
        keyword_extractor="rake",
        keywords_per_query=8,
    )
    assert config.keyword_extractor == "rake"
    assert config.query_constructor == "direct"  # Inferred default


def test_backward_compatible_tfidf_extractor():
    """Test that legacy tfidf config works unchanged (infers direct constructor)."""
    config = ReverseSearchConfig(
        keyword_extractor="tfidf",
        keywords_per_query=10,
    )
    assert config.keyword_extractor == "tfidf"
    assert config.query_constructor == "direct"  # Inferred default


# ==============================================================================
# Tests for new hybrid configuration
# ==============================================================================


def test_hybrid_config_yake_with_llm_constructor():
    """Test hybrid config: yake extraction + LLM construction."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="llm",
        keywords_per_query=10,
    )
    assert config.keyword_extractor == "yake"
    assert config.query_constructor == "llm"
    assert config.keywords_per_query == 10


def test_hybrid_config_rake_with_llm_constructor():
    """Test hybrid config: rake extraction + LLM construction."""
    config = ReverseSearchConfig(
        keyword_extractor="rake",
        query_constructor="llm",
    )
    assert config.keyword_extractor == "rake"
    assert config.query_constructor == "llm"


def test_hybrid_config_tfidf_with_llm_constructor():
    """Test hybrid config: tfidf extraction + LLM construction."""
    config = ReverseSearchConfig(
        keyword_extractor="tfidf",
        query_constructor="llm",
    )
    assert config.keyword_extractor == "tfidf"
    assert config.query_constructor == "llm"


# ==============================================================================
# Tests for invalid combinations
# ==============================================================================


def test_invalid_combo_none_extractor_with_direct_constructor():
    """Test that extractor='none' + constructor='direct' raises ConfigurationError."""
    with pytest.raises(ConfigurationError) as exc_info:
        ReverseSearchConfig(
            keyword_extractor="none",
            query_constructor="direct",
        )
    error = exc_info.value
    # Check error message and context
    assert "Invalid query generation configuration" in error.message
    assert error.context["keyword_extractor"] == "none"
    assert error.context["query_constructor"] == "direct"
    assert "issue" in error.context
    assert (
        "keyword_extractor='none' requires query_constructor='llm'"
        in error.context["issue"]
    )
    assert "suggestion" in error.context
    # Check remediation suggestion is helpful
    assert "query_constructor='llm'" in error.context["suggestion"]


def test_invalid_combo_llm_constructor_without_llm_config():
    """Test that constructor='llm' requires non-empty llm_query_config."""
    with pytest.raises(ConfigurationError) as exc_info:
        ReverseSearchConfig(
            keyword_extractor="none",
            query_constructor="llm",
            llm_query_config={},  # Empty config
        )
    error = exc_info.value
    assert "llm_query_config" in error.message.lower()
    assert error.context["query_constructor"] == "llm"
    assert "suggestion" in error.context
    # Check remediation suggestion includes example
    assert "model" in error.context["suggestion"]


# ==============================================================================
# Tests for valid new configurations
# ==============================================================================


def test_valid_config_none_extractor_with_llm_constructor():
    """Test valid config: extractor='none' + constructor='llm'."""
    config = ReverseSearchConfig(
        keyword_extractor="none",
        query_constructor="llm",
        llm_query_config={"model": "openai:gpt-4o-mini", "temperature": 0.7},
    )
    assert config.keyword_extractor == "none"
    assert config.query_constructor == "llm"
    assert config.llm_query_config["model"] == "openai:gpt-4o-mini"


def test_valid_config_explicit_direct_constructor():
    """Test valid config with explicit direct constructor."""
    config = ReverseSearchConfig(
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=7,
    )
    assert config.keyword_extractor == "yake"
    assert config.query_constructor == "direct"
    assert config.keywords_per_query == 7


# ==============================================================================
# Tests for config serialization
# ==============================================================================


def test_config_serialization_round_trip():
    """Test that config can be serialized to dict and back."""
    config = ReverseSearchConfig(
        coverage_target=0.90,
        keyword_extractor="yake",
        query_constructor="direct",
        keywords_per_query=8,
    )
    # Serialize to dict
    config_dict = config.model_dump()
    assert config_dict["keyword_extractor"] == "yake"
    assert config_dict["query_constructor"] == "direct"
    assert config_dict["keywords_per_query"] == 8

    # Deserialize from dict
    config_restored = ReverseSearchConfig(**config_dict)
    assert config_restored.keyword_extractor == "yake"
    assert config_restored.query_constructor == "direct"
    assert config_restored.keywords_per_query == 8


def test_config_serialization_with_llm_constructor():
    """Test serialization of config with LLM constructor."""
    config = ReverseSearchConfig(
        keyword_extractor="none",
        query_constructor="llm",
        llm_query_config={
            "model": "anthropic:claude-3-sonnet",
            "temperature": 0.5,
        },
    )
    config_dict = config.model_dump()
    assert config_dict["keyword_extractor"] == "none"
    assert config_dict["query_constructor"] == "llm"
    assert config_dict["llm_query_config"]["model"] == "anthropic:claude-3-sonnet"

    # Round-trip
    config_restored = ReverseSearchConfig(**config_dict)
    assert config_restored.keyword_extractor == "none"
    assert config_restored.query_constructor == "llm"
    assert config_restored.llm_query_config["model"] == "anthropic:claude-3-sonnet"


def test_config_serialization_preserves_query_construction_config():
    """Test that query_construction_config is preserved during serialization."""
    custom_construction_config = {
        "enable_fallback": False,
        "include_scores_in_prompt": False,
        "max_keywords_for_llm": 5,
    }
    config = ReverseSearchConfig(
        query_construction_config=custom_construction_config,
    )
    config_dict = config.model_dump()
    assert config_dict["query_construction_config"] == custom_construction_config

    # Round-trip
    config_restored = ReverseSearchConfig(**config_dict)
    assert config_restored.query_construction_config == custom_construction_config


# ==============================================================================
# Tests for error message clarity
# ==============================================================================


def test_error_message_includes_remediation_for_none_direct():
    """Test that error for none+direct includes specific remediation steps."""
    with pytest.raises(ConfigurationError) as exc_info:
        ReverseSearchConfig(
            keyword_extractor="none",
            query_constructor="direct",
        )
    error = exc_info.value
    # Check error message structure
    assert error.message
    assert error.context
    assert "issue" in error.context
    assert "suggestion" in error.context
    # Check remediation is actionable
    suggestion = error.context["suggestion"]
    assert "query_constructor='llm'" in suggestion
    assert "yake" in suggestion or "rake" in suggestion or "tfidf" in suggestion


def test_error_message_includes_remediation_for_empty_llm_config():
    """Test that error for empty llm_config includes example configuration."""
    with pytest.raises(ConfigurationError) as exc_info:
        ReverseSearchConfig(
            keyword_extractor="none",
            query_constructor="llm",
            llm_query_config={},
        )
    error = exc_info.value
    suggestion = error.context["suggestion"]
    # Check suggestion includes concrete example
    assert "model" in suggestion
    assert "openai" in suggestion or "gpt" in suggestion or "claude" in suggestion
    assert "temperature" in suggestion


# ==============================================================================
# Tests for field validation
# ==============================================================================


def test_query_construction_config_custom_values():
    """Test that query_construction_config accepts custom values."""
    custom_config = {
        "enable_fallback": False,
        "include_scores_in_prompt": True,
        "max_keywords_for_llm": 15,
        "custom_field": "custom_value",
    }
    config = ReverseSearchConfig(query_construction_config=custom_config)
    assert config.query_construction_config == custom_config


def test_llm_query_config_preserves_defaults_when_not_overridden():
    """Test that llm_query_config default factory is called correctly."""
    config = ReverseSearchConfig()
    # Should have default values
    assert config.llm_query_config["model"] == "openai:gpt-4o-mini"
    assert config.llm_query_config["temperature"] == 0.7
    assert config.llm_query_config["max_queries_per_resource"] == 1


def test_llm_query_config_allows_partial_override():
    """Test that llm_query_config can be partially overridden."""
    config = ReverseSearchConfig(
        llm_query_config={
            "model": "anthropic:claude-3-opus",
            "temperature": 0.3,
        }
    )
    # Overridden values
    assert config.llm_query_config["model"] == "anthropic:claude-3-opus"
    assert config.llm_query_config["temperature"] == 0.3
    # Note: With dict override, only provided fields are present
    # (not merged with defaults)


# ==============================================================================
# Tests for backward compatibility scenarios
# ==============================================================================


def test_backward_compat_minimal_config():
    """Test that minimal legacy config still works."""
    config = ReverseSearchConfig()
    # All defaults should be set
    assert config.keyword_extractor == "yake"
    assert config.query_constructor == "direct"
    assert config.coverage_target == 0.95
    assert config.max_queries == 100


def test_backward_compat_config_from_dict():
    """Test that config can be created from old dict format."""
    old_config_dict = {
        "coverage_target": 0.90,
        "keyword_extractor": "rake",
        "keywords_per_query": 5,
        # No query_constructor field (should infer "direct")
    }
    config = ReverseSearchConfig(**old_config_dict)
    assert config.keyword_extractor == "rake"
    assert config.query_constructor == "direct"  # Inferred
    assert config.keywords_per_query == 5


def test_backward_compat_llm_extractor_from_dict():
    """Test that old llm extractor config from dict is migrated."""
    old_config_dict = {
        "keyword_extractor": "llm",
        "sort_by": "date_desc",
        "llm_query_config": {
            "model": "anthropic:claude-3-sonnet",
            "temperature": 0.5,
        },
    }
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        config = ReverseSearchConfig(**old_config_dict)
        # Check migration
        assert config.keyword_extractor == "none"
        assert config.query_constructor == "llm"
        assert config.llm_query_config["model"] == "anthropic:claude-3-sonnet"
        # Check warning
        assert len(w) == 1
        assert issubclass(w[0].category, DeprecationWarning)
