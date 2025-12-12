"""Tests for simplified IfetcherConfig settings module."""

import tempfile
from pathlib import Path
import pytest
from interaction_finder.settings import IfetcherConfig, sanitize_topic_for_filename


class TestSanitizeTopicForFilename:
    """Test topic string sanitization for filenames."""

    def test_simple_topic(self):
        """Test basic topic sanitization."""
        assert sanitize_topic_for_filename("cancer") == "cancer"

    def test_multi_word_topic(self):
        """Test multi-word topics become hyphenated."""
        assert (
            sanitize_topic_for_filename("pulmonary arterial hypertension")
            == "pulmonary-arterial-hypertension"
        )

    def test_mixed_case(self):
        """Test that mixed case is lowercased."""
        assert sanitize_topic_for_filename("BRCA1 Gene") == "brca1-gene"

    def test_special_characters_removed(self):
        """Test that special characters are removed."""
        assert (
            sanitize_topic_for_filename("p53 (tumor suppressor)")
            == "p53-tumor-suppressor"
        )
        assert sanitize_topic_for_filename("gene/protein") == "geneprotein"

    def test_underscores_become_hyphens(self):
        """Test that underscores become hyphens."""
        assert (
            sanitize_topic_for_filename("cell_signaling_pathway")
            == "cell-signaling-pathway"
        )

    def test_multiple_spaces_collapsed(self):
        """Test that multiple spaces collapse to single hyphen."""
        assert sanitize_topic_for_filename("gene   expression") == "gene-expression"

    def test_leading_trailing_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        assert sanitize_topic_for_filename("  cancer research  ") == "cancer-research"

    def test_empty_string(self):
        """Test that empty string returns fallback."""
        assert sanitize_topic_for_filename("") == "output"
        assert sanitize_topic_for_filename("   ") == "output"

    def test_only_special_chars(self):
        """Test that string with only special chars returns fallback."""
        assert sanitize_topic_for_filename("!@#$%") == "output"

    def test_truncation(self):
        """Test that long topics are truncated."""
        long_topic = "a" * 100
        result = sanitize_topic_for_filename(long_topic)
        assert len(result) <= 80

    def test_truncation_no_trailing_hyphen(self):
        """Test that truncation doesn't leave trailing hyphen."""
        # Create a topic that will have a hyphen near the truncation point
        topic = "word " * 20  # Creates "word-word-word-..." pattern
        result = sanitize_topic_for_filename(topic, max_length=20)
        assert not result.endswith("-")
        assert len(result) <= 20


class TestConfigLoading:
    """Test configuration loading from TOML files."""

    def test_load_minimal_config(self):
        """Test loading a minimal configuration file."""
        toml_content = """
        [output]
        cache = "test_cache"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            # Verify loaded values
            assert config.output.cache == "test_cache"
            # Check defaults
            assert config.output.path == "{topic}.json"

            Path(f.name).unlink()

    def test_load_with_agents(self):
        """Test loading configuration with agent specs."""
        toml_content = """
        [agents.default]
        llm = "openai:gpt-4"
        retries = 3
        instrument = false

        [agents.critic]
        llm = "anthropic:claude-3-5-sonnet-20241022"
        expertise = "biological research"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            # Verify agent configs
            assert "default" in config.agents
            assert config.agents["default"].llm == "openai:gpt-4"
            assert config.agents["default"].retries == 3
            assert config.agents["default"].instrument is False

            assert "critic" in config.agents
            assert config.agents["critic"].llm == "anthropic:claude-3-5-sonnet-20241022"
            assert config.agents["critic"].expertise == "biological research"

            Path(f.name).unlink()

    def test_default_config(self):
        """Test creating config with no file (all defaults)."""
        config = IfetcherConfig()

        # Check defaults are set
        assert config.output.cache == "cache"
        assert config.output.path == "{topic}.json"
        assert config.agents == {}
        assert config.modes == {}


class TestConfigOverrides:
    """Test configuration override system."""

    def test_simple_overrides(self):
        """Test simple dotted-key overrides."""
        toml_content = """
        [output]
        cache = "original_cache"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            overrides = {"output.cache": "overridden_cache"}
            config = IfetcherConfig.from_path(f.name, overrides=overrides)

            assert config.output.cache == "overridden_cache"

            Path(f.name).unlink()

    def test_nested_overrides(self):
        """Test nested dotted-key overrides."""
        toml_content = """
        [agents.default]
        llm = "openai:gpt-4"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            overrides = {
                "agents.default.llm": "anthropic:claude-3-5-sonnet-20241022",
                "agents.default.retries": 5,
            }
            config = IfetcherConfig.from_path(f.name, overrides=overrides)

            assert (
                config.agents["default"].llm == "anthropic:claude-3-5-sonnet-20241022"
            )
            assert config.agents["default"].retries == 5

            Path(f.name).unlink()


class TestModeSpecificConfig:
    """Test mode-specific configuration overrides."""

    def test_mode_overrides_flat(self):
        """Test mode-specific overrides with flat dotted keys."""
        toml_content = """
        [output]
        cache = "default_cache"

        [modes.development]
        "output.cache" = "dev_cache"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            # Load without mode
            config_default = IfetcherConfig.from_path(f.name)
            assert config_default.output.cache == "default_cache"

            # Load with mode
            config_dev = IfetcherConfig.from_path(f.name, mode="development")
            assert config_dev.output.cache == "dev_cache"

            Path(f.name).unlink()

    def test_mode_overrides_nested(self):
        """Test mode-specific overrides with nested structure."""
        toml_content = """
        [output]
        cache = "default_cache"

        [modes.development.output]
        cache = "dev_cache"
        path = "dev_runs/{term}"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name, mode="development")
            assert config.output.cache == "dev_cache"
            assert config.output.path == "dev_runs/{term}"

            Path(f.name).unlink()


class TestPathResolution:
    """Test path resolution and formatting."""

    def test_abspath_relative(self):
        """Test absolute path resolution for relative paths."""
        toml_content = """
        [output]
        cache = "cache"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            config_dir = Path(f.name).parent

            config = IfetcherConfig.from_path(f.name)

            # Test relative path resolution
            cache_path = config.abspath("cache")
            assert cache_path.is_absolute()
            assert cache_path == (config_dir / "cache").resolve()

            Path(f.name).unlink()

    def test_abspath_with_template(self):
        """Test absolute path resolution with template formatting."""
        toml_content = """
        [output]
        path = "runs/{term}"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()
            config_dir = Path(f.name).parent

            config = IfetcherConfig.from_path(f.name)

            # Test template formatting
            output_path = config.abspath("runs/{term}", term="BRCA1")
            assert output_path.is_absolute()
            assert output_path == (config_dir / "runs" / "BRCA1").resolve()

            Path(f.name).unlink()

    def test_abspath_absolute_unchanged(self):
        """Test that absolute paths are returned unchanged."""
        config = IfetcherConfig()

        absolute_path = Path("/tmp/test")
        result = config.abspath(absolute_path)

        assert result == absolute_path


class TestValidation:
    """Test configuration validation."""

    def test_relative_path_validation(self):
        """Test that output paths must be relative."""
        toml_content = """
        [output]
        cache = "/absolute/path"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            with pytest.raises(Exception):  # Pydantic ValidationError
                IfetcherConfig.from_path(f.name)

            Path(f.name).unlink()

    def test_agent_retries_range(self):
        """Test that agent retries are validated at config creation."""
        # Validation should fail immediately when creating the config
        with pytest.raises(Exception):  # Pydantic ValidationError
            IfetcherConfig(
                agents={
                    "extraction": {
                        "judge": {"retries": 100},  # Too high, max is 10
                    }
                }
            )


class TestApplyOverrides:
    """Test the apply_overrides static method."""

    def test_simple_override(self):
        """Test applying simple override."""
        data = {"output": {"cache": "old"}}
        overrides = {"output.cache": "new"}

        result = IfetcherConfig.apply_overrides(data, overrides)

        assert result["output"]["cache"] == "new"

    def test_deep_nested_override(self):
        """Test applying deeply nested override."""
        data = {"agents": {}}
        overrides = {"agents.default.llm": "openai:gpt-4"}

        result = IfetcherConfig.apply_overrides(data, overrides)

        assert result["agents"]["default"]["llm"] == "openai:gpt-4"

    def test_comma_separated_list_override(self):
        """Test that comma-separated values become lists."""
        data = {}
        overrides = {"some.list": "a,b,c"}

        result = IfetcherConfig.apply_overrides(data, overrides)

        assert result["some"]["list"] == ["a", "b", "c"]

    def test_array_index_override(self):
        """Test overriding array elements by index."""
        data = {"items": []}
        overrides = {"items[0]": "first", "items[1]": "second"}

        result = IfetcherConfig.apply_overrides(data, overrides)

        assert result["items"] == ["first", "second"]
