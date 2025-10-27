"""
Integration tests for CLI version invocation.

Tests that --ver 1, 2, and 3 all successfully invoke their respective
pipelines without parameter mismatch errors.

These tests focus on parameter compatibility, not full pipeline execution.
They use --dry-run to verify that the CLI correctly dispatches to each version
with the appropriate parameters.
"""

import pytest
from pathlib import Path
from typer.testing import CliRunner

from interaction_finder.cli import app

runner = CliRunner()


@pytest.fixture
def mock_config_file(tmp_path):
    """Create a minimal config file for testing."""
    config_content = """
[task]
target = ["gene"]
relation = "causes"

[paths]
training_data = "test_data/{term}.jsonl"
output = "test_output/{term}.jsonl"
cache_dir = ".cache"

[agents._]
llm = "openai:gpt-4o-mini"
"""
    config_path = tmp_path / "config.toml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def mock_training_data(tmp_path):
    """Create minimal training data file in the expected location."""
    # Config says training_data = "test_data/{term}.jsonl"
    # But config paths are resolved relative to config file location (tmp_path)
    data_path = tmp_path / "test_data"
    data_path.mkdir()
    gene_file = data_path / "BRCA1.jsonl"
    gene_file.write_text('{"url": "https://example.com/doc1"}\n')
    return gene_file


def test_v1_pipeline_invocation_with_dry_run(mock_config_file, mock_training_data):
    """Test that V1 pipeline invocation works with --dry-run (no TypeError)."""
    # Use --source to provide the exact file path instead of relying on path resolution
    result = runner.invoke(
        app,
        [
            "extract",
            "-t",
            "BRCA1",
            "--ver",
            "1",
            "--config",
            str(mock_config_file),
            "--source",
            str(mock_training_data),
            "--dry-run",
        ],
        catch_exceptions=False,
    )

    # Should succeed without TypeError from parameter mismatch
    # Dry run displays processing plan and succeeds
    assert result.exit_code == 0, f"CLI failed: {result.stdout}"
    assert "Processing Plan" in result.stdout


def test_v2_pipeline_invocation_with_dry_run(mock_config_file, mock_training_data):
    """Test that V2 pipeline invocation works with --dry-run (no TypeError)."""
    result = runner.invoke(
        app,
        [
            "extract",
            "-t",
            "BRCA1",
            "--ver",
            "2",
            "--config",
            str(mock_config_file),
            "--source",
            str(mock_training_data),
            "--dry-run",
        ],
        catch_exceptions=False,
    )

    # Should succeed without TypeError from parameter mismatch
    assert result.exit_code == 0, f"CLI failed: {result.stdout}"
    assert "Processing Plan" in result.stdout


def test_v3_pipeline_invocation_with_dry_run(mock_config_file, mock_training_data):
    """Test that V3 pipeline invocation works with --dry-run (no TypeError)."""
    result = runner.invoke(
        app,
        [
            "extract",
            "-t",
            "BRCA1",
            "--ver",
            "3",
            "--config",
            str(mock_config_file),
            "--source",
            str(mock_training_data),
            "--dry-run",
        ],
        catch_exceptions=False,
    )

    # Should succeed without TypeError from parameter mismatch
    assert result.exit_code == 0, f"CLI failed: {result.stdout}"
    assert "Processing Plan" in result.stdout


def test_v1_rejects_checkpoint_parameter(mock_config_file, mock_training_data):
    """Test that V1 properly rejects --checkpoint flag."""

    result = runner.invoke(
        app,
        [
            "extract",
            "-t",
            "BRCA1",
            "--ver",
            "1",
            "--checkpoint",
            "test.json",
            "--config",
            str(mock_config_file),
        ],
    )

    # Should fail with clear error message
    assert result.exit_code == 1
    assert "checkpoint is only supported with --ver 3" in result.stdout


def test_v2_rejects_checkpoint_parameter(mock_config_file, mock_training_data):
    """Test that V2 properly rejects --checkpoint flag."""

    result = runner.invoke(
        app,
        [
            "extract",
            "-t",
            "BRCA1",
            "--ver",
            "2",
            "--checkpoint",
            "test.json",
            "--config",
            str(mock_config_file),
        ],
    )

    # Should fail with clear error message
    assert result.exit_code == 1
    assert "checkpoint is only supported with --ver 3" in result.stdout


def test_v3_accepts_checkpoint_parameter(
    mock_config_file, mock_training_data, tmp_path
):
    """Test that V3 accepts --checkpoint flag with --dry-run."""

    checkpoint_path = tmp_path / "checkpoint.json"

    result = runner.invoke(
        app,
        [
            "extract",
            "-t",
            "BRCA1",
            "--ver",
            "3",
            "--checkpoint",
            str(checkpoint_path),
            "--config",
            str(mock_config_file),
            "--source",
            str(mock_training_data),
            "--dry-run",
        ],
        catch_exceptions=False,
    )

    # Should succeed (dry-run exits before checkpoint validation)
    assert result.exit_code == 0, f"CLI failed: {result.stdout}"
    assert "Processing Plan" in result.stdout
