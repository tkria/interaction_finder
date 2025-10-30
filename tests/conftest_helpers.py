"""Conditional test execution using taint propagation through symbol dependencies.

Provides a single, simple entry point for pytest conftest files:
    skip_unless_tainted(module_name, is_base=False)

Taint Model:
    1. Modified file → file is tainted
    2. Tainted file → symbols it defines are tainted
    3. Tainted symbol → files referencing it are tainted (via AST + ripgrep)
    4. Repeat until fixed point → test file is tainted → run test

Base Module:
    The "base" pseudo-module (src/interaction_finder/*.py) has special semantics:
    tests run if ANY modification (base or submodule) propagates taint to base files.
"""

import ast
import os
import subprocess
from pathlib import Path
from typing import NamedTuple


class TaintAnalysis(NamedTuple):
    """Results of taint propagation analysis.

    Attributes:
        tainted_files: set[Path] — all files that are tainted
        tainted_symbols: set[str] — all symbols that are tainted
        initial_modifications: set[Path] — originally modified files (source of taint)
    """

    tainted_files: set[Path]
    tainted_symbols: set[str]
    initial_modifications: set[Path]


def compute_taint(
    source_directory: Path, test_directory: Path, immediate_only: bool = False
) -> TaintAnalysis:
    """Compute taint propagation from modified source files.

    Taint propagates through the following chain:
    1. Modified files are initially tainted
    2. Symbols defined in tainted files become tainted
    3. Files referencing tainted symbols become tainted
    4. This continues until no new files are tainted (fixed point)

    Parameters:
        source_directory: Path — source code directory to track modifications
        test_directory: Path — test directory to check for taint
        immediate_only: bool — if True, only track immediate children of source_directory

    Returns:
        TaintAnalysis containing all tainted files and symbols
    """
    # Get initially modified files (seeds for taint propagation)
    modified_files = get_modified_files(source_directory, immediate_only=immediate_only)
    if not modified_files:
        return TaintAnalysis(
            tainted_files=set(), tainted_symbols=set(), initial_modifications=set()
        )
    # Initialize taint sets
    tainted_files = set(modified_files)
    tainted_symbols = set()
    # Extract symbols from initially modified files
    for file_path in modified_files:
        tainted_symbols.update(get_defined_symbols(file_path))
    # Propagate taint through symbol references until fixed point
    # Use both source and test directories for propagation
    search_roots = [source_directory, test_directory]
    previous_tainted_count = 0
    # Iterate until no new tainted files are discovered (fixed point)
    while len(tainted_files) != previous_tainted_count:
        previous_tainted_count = len(tainted_files)
        # Find files referencing tainted symbols
        for search_root in search_roots:
            newly_tainted = find_files_referencing_symbols(tainted_symbols, search_root)
            # Add newly discovered tainted files
            for file_path in newly_tainted:
                if file_path not in tainted_files:
                    tainted_files.add(file_path)
                    # Extract symbols from newly tainted files for next iteration
                    tainted_symbols.update(get_defined_symbols(file_path))
    return TaintAnalysis(
        tainted_files=tainted_files,
        tainted_symbols=tainted_symbols,
        initial_modifications=set(modified_files),
    )


def is_test_tainted(test_file: Path, taint_analysis: TaintAnalysis) -> bool:
    """Check if a specific test file is tainted.

    Parameters:
        test_file: Path — test file to check
        taint_analysis: TaintAnalysis — precomputed taint information

    Returns:
        True if the test file is tainted, False otherwise
    """
    return test_file in taint_analysis.tainted_files


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


def get_defined_symbols(file_path: Path) -> set[str]:
    """Extract all top-level symbols defined in a Python file using AST.

    Parameters:
        file_path: Path — path to Python source file

    Returns:
        Set of symbol names (classes, functions, constants) defined in the file
    """
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (SyntaxError, OSError, UnicodeDecodeError):
        # File may not be valid Python or readable; return empty set
        return set()
    # Extract top-level definitions
    symbols = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.Assign):
            # Handle variable assignments: VAR = value
            for target in node.targets:
                if isinstance(target, ast.Name):
                    symbols.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            # Handle annotated assignments: VAR: type = value
            if isinstance(node.target, ast.Name):
                symbols.add(node.target.id)
    return symbols


def get_modified_files(base_path: Path, immediate_only: bool = False) -> list[Path]:
    """Get list of modified Python files in a directory according to git.

    Includes both staged and unstaged modifications, but not untracked files.

    Parameters:
        base_path: Path — directory to check for modifications
        immediate_only: bool — if True, only include immediate children (not subdirectories)

    Returns:
        List of absolute paths to modified .py files
    """
    try:
        # Get both staged and unstaged modifications
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD", str(base_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        # Parse output and filter for Python files
        modified_files = []
        for line in result.stdout.strip().split("\n"):
            if line and line.endswith(".py"):
                file_path = Path(line)
                if file_path.is_absolute():
                    abs_path = file_path
                else:
                    # Make relative paths absolute from repo root
                    abs_path = (Path.cwd() / file_path).resolve()
                # If immediate_only is True, only include direct children
                if immediate_only:
                    try:
                        # Check if file is an immediate child (parent is base_path)
                        if abs_path.parent == base_path.resolve():
                            modified_files.append(abs_path)
                    except (ValueError, AttributeError):
                        continue
                else:
                    modified_files.append(abs_path)
        return modified_files
    except (subprocess.SubprocessError, FileNotFoundError):
        # Git command failed; return empty list to be safe
        return []


def find_files_referencing_symbols(
    symbols: set[str], search_directory: Path
) -> set[Path]:
    """Find Python files that reference any of the given symbols using ripgrep.

    Uses ripgrep for fast searching across all Python files in the directory.

    Parameters:
        symbols: set[str] — symbol names to search for
        search_directory: Path — directory to search within

    Returns:
        Set of absolute paths to files that reference any of the symbols
    """
    if not symbols:
        return set()
    # Use ripgrep with word boundary matching to avoid false positives
    # Build pattern: \b(symbol1|symbol2|...)\b
    pattern = r"\b(" + "|".join(symbols) + r")\b"
    try:
        result = subprocess.run(
            [
                "rg",
                "--files-with-matches",
                "--type",
                "py",
                "--engine",
                "auto",
                pattern,
                str(search_directory),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Parse output (one file per line)
        if result.returncode == 0:
            files = {
                Path(line.strip()) for line in result.stdout.strip().split("\n") if line
            }
            return files
        else:
            # No matches found (exit code 1) or error
            return set()
    except (subprocess.SubprocessError, FileNotFoundError):
        # Ripgrep not available or failed; fall back to conservative behavior
        return set()


def should_run_tests_based_on_taint(
    test_directory: Path, source_directory: Path, immediate_only: bool = False
) -> bool:
    """Determine if tests should run based on taint propagation.

    Tests run if:
    1. TEST_ALL environment variable is set, OR
    2. Any test file in test_directory is tainted through dependency chains

    Parameters:
        test_directory: Path — directory containing test files
        source_directory: Path — source module directory to track
        immediate_only: bool — if True, only track immediate children of source_directory

    Returns:
        True if tests should run, False to skip
    """
    # Check for TEST_ALL environment variable
    if os.environ.get("TEST_ALL"):
        return True
    # Compute taint propagation
    taint_analysis = compute_taint(
        source_directory, test_directory, immediate_only=immediate_only
    )
    # If no initial modifications, nothing is tainted
    if not taint_analysis.initial_modifications:
        return False
    # Check if any test files in the test directory are tainted
    # Note: tainted_files includes files from both source and test directories
    for tainted_file in taint_analysis.tainted_files:
        try:
            # Check if this tainted file is in the test directory
            if tainted_file.is_relative_to(test_directory):
                return True
        except (ValueError, AttributeError):
            # is_relative_to can raise ValueError if paths don't share a base
            continue
    # No test files are tainted
    return False


def should_run_tests_based_on_dependencies(
    test_directory: Path, source_directory: Path
) -> bool:
    """Legacy function for backward compatibility.

    Deprecated: Use should_run_tests_based_on_taint instead.
    This function now delegates to the taint-based implementation.
    """
    return should_run_tests_based_on_taint(test_directory, source_directory)


def should_run_base_tests(test_directory: Path, source_directory: Path) -> bool:
    """Determine if base module tests should run based on taint propagation.

    Base tests run if:
    1. TEST_ALL environment variable is set, OR
    2. Any immediate child of source_directory (base source files) is tainted

    Taint can originate from ANYWHERE in the codebase (base files or submodules).
    This allows submodule changes that affect base files to trigger base tests.

    Parameters:
        test_directory: Path — directory containing base test files
        source_directory: Path — source module root directory (not just immediate files)

    Returns:
        True if base tests should run, False to skip
    """
    # Check for TEST_ALL environment variable
    if os.environ.get("TEST_ALL"):
        return True
    # Compute taint from ALL modifications in the entire source tree
    # (not just immediate files - we want to catch submodule changes too)
    taint_analysis = compute_taint(
        source_directory, test_directory, immediate_only=False
    )
    # If no tainted files at all, nothing to run
    if not taint_analysis.tainted_files:
        return False
    # Check if any IMMEDIATE source files (base module files) are tainted
    # This is the key: we propagate taint from everywhere, but only care if
    # it reaches the base-level source files
    immediate_source_files = [f for f in source_directory.glob("*.py") if f.is_file()]
    for base_file in immediate_source_files:
        if base_file in taint_analysis.tainted_files:
            return True
    # No base source files are tainted, but check if any base test files are tainted
    # (This handles cases where tests import from submodules directly)
    for tainted_file in taint_analysis.tainted_files:
        try:
            # Check if this tainted file is a base-level test file
            if (
                tainted_file.is_relative_to(test_directory)
                and tainted_file.parent == test_directory
            ):
                return True
        except (ValueError, AttributeError):
            continue
    # No base files (source or test) are tainted
    return False


def get_skip_message(module_name: str) -> str:
    """Generate skip message for a specific module.

    Parameters:
        module_name: str — name of the module (e.g., "fetcher", "search")

    Returns:
        Formatted skip message
    """
    return (
        f"Skipping {module_name} tests (TEST_ALL not set and no tests tainted "
        f"through dependencies on src/interaction_finder/{module_name}/)"
    )


# Public API: single entry point for all pytest conftest files
def skip_unless_tainted(module_name: str, is_base: bool = False):
    """Factory function creating a pytest_collection_modifyitems hook.

    This is the main entry point for conditional test execution. Use in conftest.py:

        from tests.conftest_helpers import skip_unless_tainted
        pytest_collection_modifyitems = skip_unless_tainted("search")

    Or for the base pseudo-module:

        pytest_collection_modifyitems = skip_unless_tainted("base", is_base=True)

    Parameters:
        module_name: str — module name for skip messages (e.g., "search", "fetcher")
        is_base: bool — if True, use base module semantics (taint from anywhere)

    Returns:
        Callable compatible with pytest_collection_modifyitems hook
    """

    # Capture the conftest path at factory creation time (when skip_unless_tainted is called)
    import inspect

    factory_frame = inspect.currentframe().f_back
    conftest_file = Path(factory_frame.f_globals["__file__"])
    test_dir = conftest_file.parent
    # Find repo root: walk up until we find a directory containing 'tests' and 'src'
    # This handles both tests/conftest.py (test_dir=tests) and tests/fetcher/conftest.py (test_dir=tests/fetcher)
    repo_root = test_dir
    while repo_root.name != "" and not (
        (repo_root / "tests").exists() and (repo_root / "src").exists()
    ):
        repo_root = repo_root.parent
    src_root = repo_root / "src" / "interaction_finder"

    def pytest_collection_modifyitems(config, items):
        """Skip tests unless tainted through symbol dependencies."""
        # Import pytest here to avoid issues when module is imported outside pytest
        import pytest

        # Determine if tests should run
        if is_base:
            # Base module: check if taint reaches base-level files
            should_run = os.environ.get("TEST_ALL") or _base_files_tainted(
                test_dir, src_root
            )
        else:
            # Submodule: standard taint propagation
            module_src = src_root / module_name
            should_run = os.environ.get("TEST_ALL") or _module_tests_tainted(
                test_dir, module_src
            )
        if should_run:
            return  # Run all tests
        # Skip tests from this directory
        skip_marker = pytest.mark.skip(reason=get_skip_message(module_name))
        for item in items:
            if Path(item.fspath).is_relative_to(test_dir):
                item.add_marker(skip_marker)

    return pytest_collection_modifyitems


def _module_tests_tainted(test_dir: Path, module_src: Path) -> bool:
    """Check if any test in test_dir is tainted by module_src changes."""
    taint = compute_taint(module_src, test_dir, immediate_only=False)
    if not taint.tainted_files:
        return False
    # Check if any test files in test_dir are tainted
    for tainted_file in taint.tainted_files:
        try:
            if tainted_file.is_relative_to(test_dir):
                return True
        except (ValueError, AttributeError):
            continue
    return False


def _base_files_tainted(test_dir: Path, src_root: Path) -> bool:
    """Check if base-level files are tainted by any modifications."""
    taint = compute_taint(src_root, test_dir, immediate_only=False)
    if not taint.tainted_files:
        return False
    # Check if any immediate source files (base files) are tainted
    for base_file in src_root.glob("*.py"):
        if base_file.is_file() and base_file in taint.tainted_files:
            return True
    # Also check if any base-level test files are tainted
    for tainted_file in taint.tainted_files:
        try:
            if (
                tainted_file.is_relative_to(test_dir)
                and tainted_file.parent == test_dir
            ):
                return True
        except (ValueError, AttributeError):
            continue
    return False
