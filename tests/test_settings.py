"""
Tests for settings and configuration functionality.

Tests cover:
- IfetcherConfig loading from TOML files
- Configuration validation and defaults
- Override functionality with dotted keys
- Mode-specific configuration
- Path resolution and validation
"""

import pytest
import tempfile
import tomli
from pathlib import Path
from unittest.mock import patch, mock_open
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from interaction_finder.settings import IfetcherConfig


class TestConfigLoading:
    """Test configuration loading from TOML files."""

    def test_load_minimal_config(self):
        """Test loading a minimal configuration file."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene", form = ["name"] }
        disease = { kind = "disease", form = ["name"] }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            # Check basic structure
            assert config.task.kinds["gene"].kind == ["gene"]
            assert config.task.kinds["disease"].kind == ["disease"]

            # Check defaults
            assert config.workflow.max_loops == 5
            assert config.tools.crawl4ai.timeout == 30

            # Clean up
            Path(f.name).unlink()

    def test_load_full_config(self):
        """Test loading a comprehensive configuration."""
        toml_content = """
        researcher = ""
        training_data = "data/{term}.jsonl"
        
        [agents.gene_disease]
        llm = "gpt-4"
        expertise = "biomedical"
        retries = 3
        
        [task]
        relation = "Interaction"
        pairs = "pairs"
        context = "biological context"
        
        [task.kinds]
        gene = { kind = "gene", form = ["name", "symbol"] }
        disease = { kind = "disease", form = ["name"] }
        
        [workflow]
        max_loops = 10
        max_tokens = 100000
        
        [workflow.grouping]
        enabled = true
        min_size = 2
        max_size = 6
        
        [tools.crawl4ai]
        timeout = 60
        max_retries = 5
        
        [output]
        path = "results/{term}"
        cache = "my_cache"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            # Check loaded values
            assert config.training_data == "data/{term}.jsonl"
            assert config.agents["gene_disease"].llm == "gpt-4"
            assert config.agents["gene_disease"].retries == 3
            assert config.task.relation == "Interaction"
            assert config.task.context == "biological context"
            assert config.workflow.max_loops == 10
            assert config.workflow.max_tokens == 100000
            assert config.workflow.grouping.min_size == 2
            assert config.workflow.grouping.max_size == 6
            assert config.tools.crawl4ai.timeout == 60
            assert config.tools.crawl4ai.max_retries == 5
            assert config.output.path == "results/{term}"
            assert config.output.cache == "my_cache"

            Path(f.name).unlink()


class TestConfigOverrides:
    """Test configuration override functionality."""

    def test_simple_overrides(self):
        """Test simple dotted-key overrides."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            overrides = {
                "workflow.max_loops": 15,
                "tools.crawl4ai.timeout": 120,
                "task.relation": "CustomRelation",
            }

            config = IfetcherConfig.from_path(f.name, overrides=overrides)

            assert config.workflow.max_loops == 15
            assert config.tools.crawl4ai.timeout == 120
            assert config.task.relation == "CustomRelation"

            Path(f.name).unlink()

    def test_nested_overrides(self):
        """Test nested configuration overrides."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            overrides = {
                "workflow.grouping.min_size": 5,
                "workflow.grouping.max_size": 15,
                "agents.custom.llm": "custom-model",
            }

            config = IfetcherConfig.from_path(f.name, overrides=overrides)

            assert config.workflow.grouping.min_size == 5
            assert config.workflow.grouping.max_size == 15
            assert config.agents["custom"].llm == "custom-model"

            Path(f.name).unlink()

    def test_list_overrides(self):
        """Test overriding list values."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            overrides = {
                "tools.enabled": "tool1,tool2,tool3",
                "task.example": "ex1,ex2",
            }

            config = IfetcherConfig.from_path(f.name, overrides=overrides)

            assert config.tools.enabled == ["tool1", "tool2", "tool3"]
            assert config.task.example == ["ex1", "ex2"]

            Path(f.name).unlink()


class TestModeSpecificConfig:
    """Test mode-specific configuration functionality."""

    def test_mode_overrides_flat(self):
        """Test mode overrides in flat dotted format."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        
        [modes.test]
        "workflow.max_loops" = 20
        "tools.crawl4ai.timeout" = 90
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name, mode="test")

            assert config.workflow.max_loops == 20
            assert config.tools.crawl4ai.timeout == 90

            Path(f.name).unlink()

    def test_mode_overrides_nested(self):
        """Test mode overrides in nested format."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        
        [modes.dev.workflow]
        max_loops = 3
        max_tokens = 50000
        
        [modes.dev.tools.crawl4ai]
        timeout = 15
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name, mode="dev")

            assert config.workflow.max_loops == 3
            assert config.workflow.max_tokens == 50000
            assert config.tools.crawl4ai.timeout == 15

            Path(f.name).unlink()


class TestValidation:
    """Test configuration validation."""

    def test_researcher_validation(self):
        """Test researcher field validation."""
        toml_content = """
        researcher = "invalid_mode"
        
        [task.kinds]
        gene = { kind = "gene" }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            with pytest.raises(ValueError, match="researcher must be one of"):
                IfetcherConfig.from_path(f.name)

            Path(f.name).unlink()

    def test_kinds_validation_empty(self):
        """Test that empty kinds configuration raises error."""
        toml_content = """
        [task]
        relation = "test"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            with pytest.raises(ValueError, match="At least one kind must be defined"):
                IfetcherConfig.from_path(f.name)

            Path(f.name).unlink()

    def test_grouping_validation(self):
        """Test grouping configuration validation."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        
        [workflow.grouping]
        min_size = 10
        max_size = 5
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            with pytest.raises(ValueError, match="min_size must be <= max_size"):
                IfetcherConfig.from_path(f.name)

            Path(f.name).unlink()

    def test_output_path_validation(self):
        """Test output path validation (must be relative)."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        
        [output]
        path = "/absolute/path"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            with pytest.raises(ValueError, match="output paths must be relative"):
                IfetcherConfig.from_path(f.name)

            Path(f.name).unlink()


class TestKindsNormalization:
    """Test task kinds normalization functionality."""

    def test_kinds_array_format(self):
        """Test array format for kinds (["gene", "disease"])."""
        toml_content = """
        [task]
        kinds = ["gene", "disease"]
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            assert "gene" in config.task.kinds
            assert "disease" in config.task.kinds
            assert config.task.kinds["gene"].kind == ["gene"]
            assert config.task.kinds["disease"].kind == ["disease"]

            Path(f.name).unlink()

    def test_kinds_direct_object(self):
        """Test direct Kind object format."""
        toml_content = """
        [task.kinds]
        kind = "gene"
        form = ["name", "symbol"]
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            assert "default" in config.task.kinds
            assert config.task.kinds["default"].kind == ["gene"]
            assert config.task.kinds["default"].form == ["name", "symbol"]

            Path(f.name).unlink()

    def test_kinds_named_objects(self):
        """Test named Kind objects format."""
        toml_content = """
        [task.kinds.gene]
        kind = "gene"
        form = ["name", "symbol"]
        example = ["BRCA1", "TP53"]
        
        [task.kinds.disease]
        kind = "disease"
        form = ["name"]
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            assert config.task.kinds["gene"].kind == ["gene"]
            assert config.task.kinds["gene"].form == ["name", "symbol"]
            assert config.task.kinds["gene"].example == ["BRCA1", "TP53"]
            assert config.task.kinds["disease"].kind == ["disease"]
            assert config.task.kinds["disease"].form == ["name"]

            Path(f.name).unlink()


class TestPathResolution:
    """Test path resolution functionality."""

    def test_abspath_relative_paths(self):
        """Test absolute path resolution for relative paths."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            # Test relative path resolution
            abs_path = config.abspath("data/{term}.jsonl", term="BRCA1")
            expected_dir = Path(f.name).parent
            expected_path = expected_dir / "data" / "BRCA1.jsonl"

            assert abs_path == expected_path.resolve()

            Path(f.name).unlink()

    def test_abspath_absolute_paths(self):
        """Test that absolute paths are returned as-is."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            # Test absolute path (unchanged)
            abs_input = Path("/tmp/absolute/path.txt")
            result = config.abspath(abs_input)

            assert result == abs_input

            Path(f.name).unlink()

    def test_abspath_with_formatting(self):
        """Test path resolution with string formatting."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            # Test with multiple format parameters
            result = config.abspath(
                "runs/{mode}/{model}/{term}.json",
                mode="train",
                model="gpt-4",
                term="BRCA1",
            )

            expected_dir = Path(f.name).parent
            expected = expected_dir / "runs" / "train" / "gpt-4" / "BRCA1.json"

            assert result == expected.resolve()

            Path(f.name).unlink()


class TestUtilityMethods:
    """Test utility methods on configuration objects."""

    def test_get_kind_names(self):
        """Test getting list of kind names."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        disease = { kind = "disease" }
        protein = { kind = "protein" }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            kind_names = config.task.get_kind_names()
            assert set(kind_names) == {"gene", "disease", "protein"}

            Path(f.name).unlink()

    def test_get_complementary_kind_two_kinds(self):
        """Test getting complementary kind with two kinds."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        disease = { kind = "disease" }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            assert config.task.get_complementary_kind("gene") == "disease"
            assert config.task.get_complementary_kind("disease") == "gene"

            Path(f.name).unlink()

    def test_get_complementary_kind_one_kind(self):
        """Test getting complementary kind with one kind."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            assert config.task.get_complementary_kind("gene") == "gene"

            Path(f.name).unlink()

    def test_get_complementary_kind_multiple_kinds_error(self):
        """Test error with more than two kinds."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        disease = { kind = "disease" }
        protein = { kind = "protein" }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            with pytest.raises(ValueError, match="Cannot get complementary kind"):
                config.task.get_complementary_kind("gene")

            Path(f.name).unlink()


class TestExternalResearcherConfig:
    """Test external researcher configuration."""

    def test_cmd_string_parsing(self):
        """Test command string parsing with shlex."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        
        [tools.external_researcher]
        cmd = 'python script.py --arg "value with spaces"'
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            expected_cmd = ["python", "script.py", "--arg", "value with spaces"]
            assert config.tools.external_researcher.cmd == expected_cmd

            Path(f.name).unlink()

    def test_cmd_array_format(self):
        """Test command as array format."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        
        [tools.external_researcher]
        cmd = ["python", "script.py", "--verbose"]
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            assert config.tools.external_researcher.cmd == [
                "python",
                "script.py",
                "--verbose",
            ]

            Path(f.name).unlink()

    def test_strip_re_normalization(self):
        """Test strip_re field normalization."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        
        [tools.external_researcher]
        strip_re = "^prefix:"
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            assert config.tools.external_researcher.strip_re == ["^prefix:"]

            Path(f.name).unlink()


class TestDefaults:
    """Test default configuration values."""

    def test_minimal_config_defaults(self):
        """Test that minimal config gets proper defaults."""
        toml_content = """
        [task.kinds]
        gene = { kind = "gene" }
        """

        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write(toml_content)
            f.flush()

            config = IfetcherConfig.from_path(f.name)

            # Check critical defaults
            assert config.researcher == ""
            assert config.task.relation == "Interaction"
            assert config.task.pairs == "pairs"
            assert config.workflow.max_loops == 5
            assert config.workflow.max_tokens == 500000
            assert config.workflow.grouping.enabled == True
            assert config.workflow.grouping.min_size == 3
            assert config.workflow.grouping.max_size == 8
            assert config.tools.crawl4ai.timeout == 30
            assert config.tools.crawl4ai.max_retries == 3
            assert config.output.path == "runs/{mode}/{model}/{repeat}/{term}"
            assert config.output.cache == "cache"
            assert config.training_data == "training_data/{term}.jsonl"

            Path(f.name).unlink()
