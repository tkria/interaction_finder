"""Tests for simplified IfetcherConfig settings module."""

import tempfile
from pathlib import Path
import pytest
from interaction_finder.settings import IfetcherConfig


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
            assert config.output.path == "runs/{mode}/{model}/{repeat}/{term}"
            assert config.training_data == "training_data/{term}.jsonl"

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
        assert config.output.path == "runs/{mode}/{model}/{repeat}/{term}"
        assert config.training_data == "training_data/{term}.jsonl"
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
        """Test that agent retries are validated."""
        toml_content = """
        [agents.default]
        retries = 100  # Too high, max is 10
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            with pytest.raises(Exception):  # Pydantic ValidationError
                IfetcherConfig.from_path(f.name)

            Path(f.name).unlink()


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
