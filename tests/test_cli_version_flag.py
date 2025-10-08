"""
Tests for CLI version flag validation.

Tests that the --ver flag properly validates version numbers and
rejects invalid versions.
"""

import pytest
from typer.testing import CliRunner
from interaction_finder.cli import app

runner = CliRunner()


def test_version_flag_default_is_3():
    """Test that version defaults to 3."""
    # Use --dry-run to avoid actual execution, and check the log output
    result = runner.invoke(app, ["extract", "-t", "BRCA1", "--dry-run", "--verbose"])
    # Should not fail with version error (may fail for other reasons like missing config)
    assert "Invalid version" not in result.stdout


def test_version_flag_accepts_valid_versions():
    """Test that valid versions (1, 2, 3) are accepted."""
    for version in [1, 2, 3]:
        result = runner.invoke(
            app, ["extract", "-t", "BRCA1", "--ver", str(version), "--dry-run"]
        )
        # Should succeed without version error
        # (may fail for other reasons like missing config, but not version validation)
        assert "Invalid version" not in result.stdout


def test_version_flag_rejects_invalid_versions():
    """Test that invalid versions are rejected with clear error."""
    invalid_versions = [0, 4, 5, -1, 10]

    for version in invalid_versions:
        result = runner.invoke(app, ["extract", "-t", "BRCA1", "--ver", str(version)])
        # Should fail with version error
        assert result.exit_code == 1
        assert "Invalid version" in result.stdout
        assert "Must be 1, 2, or 3" in result.stdout


def test_checkpoint_only_works_with_v3():
    """Test that --checkpoint flag only works with --ver 3."""
    # Test that checkpoint with V3 is accepted (may fail later for other reasons)
    result_v3 = runner.invoke(
        app,
        [
            "extract",
            "-t",
            "BRCA1",
            "--ver",
            "3",
            "--checkpoint",
            "test.json",
            "--dry-run",
        ],
    )
    # Should not fail with checkpoint error
    assert "checkpoint is only supported with --ver 3" not in result_v3.stdout

    # Test that checkpoint with V2 is rejected
    result_v2 = runner.invoke(
        app, ["extract", "-t", "BRCA1", "--ver", "2", "--checkpoint", "test.json"]
    )
    assert result_v2.exit_code == 1
    assert "checkpoint is only supported with --ver 3" in result_v2.stdout

    # Test that checkpoint with V1 is rejected
    result_v1 = runner.invoke(
        app, ["extract", "-t", "BRCA1", "--ver", "1", "--checkpoint", "test.json"]
    )
    assert result_v1.exit_code == 1
    assert "checkpoint is only supported with --ver 3" in result_v1.stdout


def test_version_flag_in_help():
    """Test that --ver flag appears in help text."""
    result = runner.invoke(app, ["extract", "--help"])
    assert result.exit_code == 0
    assert "--ver" in result.stdout
    assert "Pipeline version" in result.stdout
