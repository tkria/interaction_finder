"""Shared utilities for conditional test execution based on git status.

This module provides reusable helpers for pytest conftest files that want to
conditionally skip tests based on:
1. TEST_ALL environment variable being set
2. Uncommitted modifications in specific source directories
"""

import os
import subprocess
from pathlib import Path


def should_run_tests_for_module(module_path: Path) -> bool:
    """Determine if tests should run for a specific source module.

    Tests run if:
    1. TEST_ALL environment variable is set, OR
    2. Any file under the specified module path has uncommitted modifications

    Parameters:
        module_path: Path — absolute path to the source module directory

    Returns:
        True if tests should run, False to skip
    """
    # Check for TEST_ALL environment variable
    if os.environ.get("TEST_ALL"):
        return True

    # Check for uncommitted modifications in the module directory
    try:
        # Run git status --porcelain on the module directory
        result = subprocess.run(
            ["git", "status", "--porcelain", str(module_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        # If there's any output, there are uncommitted changes
        has_changes = bool(result.stdout.strip())
        return has_changes
    except (subprocess.SubprocessError, FileNotFoundError):
        # If git command fails (not a git repo, git not available, etc.),
        # default to running tests to be safe
        return True


def get_skip_message(module_name: str) -> str:
    """Generate skip message for a specific module.

    Parameters:
        module_name: str — name of the module (e.g., "fetcher", "search")

    Returns:
        Formatted skip message
    """
    return (
        f"Skipping {module_name} tests (TEST_ALL not set and no uncommitted "
        f"changes in src/interaction_finder/{module_name}/)"
    )
