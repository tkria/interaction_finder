"""Tests for conditional test execution helpers with taint propagation."""

import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

from tests.conftest_helpers import (
    TaintAnalysis,
    compute_taint,
    find_files_referencing_symbols,
    get_defined_symbols,
    get_modified_files,
    get_skip_message,
    is_test_tainted,
    should_run_base_tests,
    should_run_tests_based_on_dependencies,
    should_run_tests_based_on_taint,
    should_run_tests_for_module,
)


class TestGetDefinedSymbols:
    """Tests for AST-based symbol extraction."""

    def test_extracts_function_definitions(self, tmp_path):
        """Should extract function names from Python files."""
        source_file = tmp_path / "module.py"
        source_file.write_text(
            """
def foo():
    pass

def bar():
    pass

async def baz():
    pass
"""
        )
        symbols = get_defined_symbols(source_file)
        assert symbols == {"foo", "bar", "baz"}

    def test_extracts_class_definitions(self, tmp_path):
        """Should extract class names from Python files."""
        source_file = tmp_path / "module.py"
        source_file.write_text(
            """
class Foo:
    pass

class Bar:
    def method(self):
        pass
"""
        )
        symbols = get_defined_symbols(source_file)
        # Only top-level symbols, not methods
        assert symbols == {"Foo", "Bar"}

    def test_extracts_variable_assignments(self, tmp_path):
        """Should extract top-level variable names."""
        source_file = tmp_path / "module.py"
        source_file.write_text(
            """
CONSTANT = 42
another_var = "hello"
x, y = 1, 2
"""
        )
        symbols = get_defined_symbols(source_file)
        assert "CONSTANT" in symbols
        assert "another_var" in symbols

    def test_extracts_annotated_assignments(self, tmp_path):
        """Should extract annotated variable assignments."""
        source_file = tmp_path / "module.py"
        source_file.write_text(
            """
FOO: int = 42
BAR: str = "hello"
"""
        )
        symbols = get_defined_symbols(source_file)
        assert symbols == {"FOO", "BAR"}

    def test_handles_syntax_errors_gracefully(self, tmp_path):
        """Should return empty set for files with syntax errors."""
        source_file = tmp_path / "broken.py"
        source_file.write_text("def foo(:\n    pass")
        symbols = get_defined_symbols(source_file)
        assert symbols == set()

    def test_handles_non_python_files(self, tmp_path):
        """Should return empty set for non-Python content."""
        source_file = tmp_path / "data.py"
        source_file.write_bytes(b"\x00\xff\xfe")
        symbols = get_defined_symbols(source_file)
        assert symbols == set()

    def test_handles_missing_files(self, tmp_path):
        """Should return empty set for non-existent files."""
        source_file = tmp_path / "missing.py"
        symbols = get_defined_symbols(source_file)
        assert symbols == set()

    def test_ignores_nested_definitions(self, tmp_path):
        """Should only extract top-level symbols, not nested ones."""
        source_file = tmp_path / "module.py"
        source_file.write_text(
            """
def outer():
    def inner():
        pass
    return inner

class Outer:
    class Inner:
        pass
"""
        )
        symbols = get_defined_symbols(source_file)
        assert symbols == {"outer", "Outer"}
        assert "inner" not in symbols
        assert "Inner" not in symbols


class TestGetModifiedFiles:
    """Tests for git diff-based modified file detection."""

    def test_returns_empty_for_no_modifications(self, tmp_path):
        """Should return empty list when no files are modified."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="", returncode=0)
            result = get_modified_files(tmp_path)
            assert result == []

    def test_filters_python_files_only(self, tmp_path):
        """Should only include .py files in results."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="src/foo.py\nsrc/bar.txt\nsrc/baz.py\n", returncode=0
            )
            result = get_modified_files(tmp_path)
            assert len(result) == 2
            assert all(str(f).endswith(".py") for f in result)

    def test_handles_git_command_failure(self, tmp_path):
        """Should return empty list if git command fails."""
        with patch("subprocess.run", side_effect=subprocess.SubprocessError):
            result = get_modified_files(tmp_path)
            assert result == []

    def test_handles_missing_git(self, tmp_path):
        """Should return empty list if git is not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = get_modified_files(tmp_path)
            assert result == []

    def test_makes_relative_paths_absolute(self, tmp_path):
        """Should convert relative paths to absolute."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="src/module.py\n", returncode=0)
            result = get_modified_files(tmp_path)
            assert len(result) == 1
            assert result[0].is_absolute()


class TestFindFilesReferencingSymbols:
    """Tests for ripgrep-based symbol reference detection."""

    def test_returns_empty_for_no_symbols(self, tmp_path):
        """Should return empty set when no symbols provided."""
        result = find_files_referencing_symbols(set(), tmp_path)
        assert result == set()

    def test_finds_files_with_symbol_references(self, tmp_path):
        """Should find files that reference the given symbols."""
        test_file = tmp_path / "test.py"
        test_file.write_text("from module import foo\nfoo()")
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout=f"{test_file}\n", returncode=0)
            result = find_files_referencing_symbols({"foo"}, tmp_path)
            assert len(result) == 1

    def test_handles_multiple_symbols(self, tmp_path):
        """Should search for all provided symbols."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="", returncode=1)  # No matches
            find_files_referencing_symbols({"foo", "bar", "baz"}, tmp_path)
            # Verify pattern includes all symbols
            call_args = mock_run.call_args[0][0]
            pattern = call_args[call_args.index("--engine") + 2]
            assert "foo" in pattern
            assert "bar" in pattern
            assert "baz" in pattern

    def test_handles_no_matches(self, tmp_path):
        """Should return empty set when no files match."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="", returncode=1)
            result = find_files_referencing_symbols({"nonexistent"}, tmp_path)
            assert result == set()

    def test_handles_ripgrep_failure(self, tmp_path):
        """Should return empty set if ripgrep fails."""
        with patch("subprocess.run", side_effect=subprocess.SubprocessError):
            result = find_files_referencing_symbols({"foo"}, tmp_path)
            assert result == set()

    def test_handles_missing_ripgrep(self, tmp_path):
        """Should return empty set if ripgrep is not available."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = find_files_referencing_symbols({"foo"}, tmp_path)
            assert result == set()


class TestShouldRunTestsBasedOnDependencies:
    """Tests for integrated symbol-based dependency tracking."""

    def test_runs_when_test_all_set(self, tmp_path, monkeypatch):
        """Should always run tests when TEST_ALL is set."""
        monkeypatch.setenv("TEST_ALL", "1")
        result = should_run_tests_based_on_dependencies(tmp_path, tmp_path)
        assert result is True

    def test_skips_when_no_modifications(self, tmp_path, monkeypatch):
        """Should skip tests when no files are modified."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        with patch("tests.conftest_helpers.get_modified_files", return_value=[]):
            result = should_run_tests_based_on_dependencies(tmp_path, tmp_path)
            assert result is False

    def test_runs_when_symbols_referenced_in_tests(self, tmp_path, monkeypatch):
        """Should run tests when modified symbols are referenced."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        modified_file = tmp_path / "src" / "module.py"
        test_file = tmp_path / "tests" / "test_module.py"
        with (
            patch(
                "tests.conftest_helpers.get_modified_files",
                return_value=[modified_file],
            ),
            patch(
                "tests.conftest_helpers.get_defined_symbols",
                return_value={"foo", "bar"},
            ),
            patch(
                "tests.conftest_helpers.find_files_referencing_symbols",
                return_value={test_file},
            ),
        ):
            result = should_run_tests_based_on_dependencies(
                tmp_path / "tests", tmp_path / "src"
            )
            assert result is True

    def test_skips_when_no_symbols_referenced(self, tmp_path, monkeypatch):
        """Should skip tests when modified symbols are not referenced."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        modified_file = tmp_path / "src" / "module.py"
        with (
            patch(
                "tests.conftest_helpers.get_modified_files",
                return_value=[modified_file],
            ),
            patch(
                "tests.conftest_helpers.get_defined_symbols",
                return_value={"foo", "bar"},
            ),
            patch(
                "tests.conftest_helpers.find_files_referencing_symbols",
                return_value=set(),
            ),
        ):
            result = should_run_tests_based_on_dependencies(
                tmp_path / "tests", tmp_path / "src"
            )
            assert result is False

    def test_skips_when_modifications_but_no_symbols(self, tmp_path, monkeypatch):
        """Should skip tests if files modified but no symbols extracted (no taint propagation)."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        modified_file = tmp_path / "src" / "module.py"
        test_dir = tmp_path / "tests"
        src_dir = tmp_path / "src"
        with (
            patch(
                "tests.conftest_helpers.get_modified_files",
                return_value=[modified_file],
            ),
            patch("tests.conftest_helpers.get_defined_symbols", return_value=set()),
            patch(
                "tests.conftest_helpers.find_files_referencing_symbols",
                return_value=set(),
            ),
        ):
            result = should_run_tests_based_on_dependencies(test_dir, src_dir)
            # With taint model: no symbols = no taint propagation = tests skipped
            assert result is False


class TestShouldRunTestsForModule:
    """Tests for legacy directory-based test execution (backward compatibility)."""

    def test_runs_when_test_all_set(self, tmp_path, monkeypatch):
        """Should run tests when TEST_ALL environment variable is set."""
        monkeypatch.setenv("TEST_ALL", "1")
        result = should_run_tests_for_module(tmp_path)
        assert result is True

    def test_runs_when_modifications_exist(self, tmp_path, monkeypatch):
        """Should run tests when modifications exist in the module."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="M src/module.py\n", returncode=0)
            result = should_run_tests_for_module(tmp_path)
            assert result is True

    def test_skips_when_no_modifications(self, tmp_path, monkeypatch):
        """Should skip tests when no modifications exist."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="", returncode=0)
            result = should_run_tests_for_module(tmp_path)
            assert result is False

    def test_handles_git_failure_safely(self, tmp_path, monkeypatch):
        """Should default to running tests when git command fails."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        with patch("subprocess.run", side_effect=subprocess.SubprocessError):
            result = should_run_tests_for_module(tmp_path)
            # Safe fallback: run tests
            assert result is True


class TestGetSkipMessage:
    """Tests for skip message generation."""

    def test_generates_correct_message(self):
        """Should generate appropriate skip message."""
        message = get_skip_message("fetcher")
        assert "fetcher" in message
        assert "TEST_ALL" in message
        assert "src/interaction_finder/fetcher" in message


class TestComputeTaint:
    """Tests for taint propagation computation."""

    def test_no_modifications_returns_empty_taint(self, tmp_path):
        """Should return empty taint when no files are modified."""
        src_dir = tmp_path / "src"
        test_dir = tmp_path / "tests"
        with patch("tests.conftest_helpers.get_modified_files", return_value=[]):
            result = compute_taint(src_dir, test_dir)
            assert result.tainted_files == set()
            assert result.tainted_symbols == set()
            assert result.initial_modifications == set()

    def test_modified_test_file_taints_itself(self, tmp_path):
        """Modified test file should be initially tainted."""
        src_dir = tmp_path / "src"
        test_dir = tmp_path / "tests"
        modified_test_file = test_dir / "test_module.py"

        # Mock get_modified_files to return different results based on directory
        def mock_get_modified(directory, immediate_only=False):
            if directory == test_dir:
                return [modified_test_file]
            return []

        with (
            patch(
                "tests.conftest_helpers.get_modified_files",
                side_effect=mock_get_modified,
            ),
            patch(
                "tests.conftest_helpers.get_defined_symbols", return_value={"test_foo"}
            ),
            patch(
                "tests.conftest_helpers.find_files_referencing_symbols",
                return_value=set(),
            ),
        ):
            result = compute_taint(src_dir, test_dir)
            assert modified_test_file in result.tainted_files
            assert "test_foo" in result.tainted_symbols
            assert modified_test_file in result.initial_modifications

    def test_modified_test_and_source_files_both_tainted(self, tmp_path):
        """Both modified test and source files should be initially tainted."""
        src_dir = tmp_path / "src"
        test_dir = tmp_path / "tests"
        modified_source_file = src_dir / "module.py"
        modified_test_file = test_dir / "test_module.py"

        # Mock get_modified_files to return different results based on directory
        def mock_get_modified(directory, immediate_only=False):
            if directory == src_dir:
                return [modified_source_file]
            elif directory == test_dir:
                return [modified_test_file]
            return []

        with (
            patch(
                "tests.conftest_helpers.get_modified_files",
                side_effect=mock_get_modified,
            ),
            patch("tests.conftest_helpers.get_defined_symbols", return_value={"foo"}),
            patch(
                "tests.conftest_helpers.find_files_referencing_symbols",
                return_value=set(),
            ),
        ):
            result = compute_taint(src_dir, test_dir)
            assert modified_source_file in result.tainted_files
            assert modified_test_file in result.tainted_files
            assert modified_source_file in result.initial_modifications
            assert modified_test_file in result.initial_modifications

    def test_modified_file_taints_itself(self, tmp_path):
        """Modified file should be initially tainted."""
        src_dir = tmp_path / "src"
        test_dir = tmp_path / "tests"
        modified_file = src_dir / "module.py"
        with (
            patch(
                "tests.conftest_helpers.get_modified_files",
                return_value=[modified_file],
            ),
            patch("tests.conftest_helpers.get_defined_symbols", return_value={"foo"}),
            patch(
                "tests.conftest_helpers.find_files_referencing_symbols",
                return_value=set(),
            ),
        ):
            result = compute_taint(src_dir, test_dir)
            assert modified_file in result.tainted_files
            assert "foo" in result.tainted_symbols
            assert modified_file in result.initial_modifications

    def test_taint_propagates_through_symbols(self, tmp_path):
        """Taint should propagate to files referencing tainted symbols."""
        src_dir = tmp_path / "src"
        test_dir = tmp_path / "tests"
        modified_file = src_dir / "module.py"
        dependent_file = src_dir / "dependent.py"
        test_file = test_dir / "test_module.py"
        # First call: find files referencing the initial symbol
        # Second call: no new references found (fixed point)
        call_count = 0

        def mock_find_references(symbols, search_root):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and search_root == src_dir:
                # First iteration: dependent file uses the symbol
                return {dependent_file}
            elif call_count == 2 and search_root == test_dir:
                # First iteration: test file uses the symbol
                return {test_file}
            else:
                # Later iterations: no new references
                return set()

        with (
            patch(
                "tests.conftest_helpers.get_modified_files",
                return_value=[modified_file],
            ),
            patch("tests.conftest_helpers.get_defined_symbols", return_value={"foo"}),
            patch(
                "tests.conftest_helpers.find_files_referencing_symbols",
                side_effect=mock_find_references,
            ),
        ):
            result = compute_taint(src_dir, test_dir)
            # All three files should be tainted
            assert modified_file in result.tainted_files
            assert dependent_file in result.tainted_files
            assert test_file in result.tainted_files

    def test_taint_reaches_fixed_point(self, tmp_path):
        """Taint propagation should stop when no new files are tainted."""
        src_dir = tmp_path / "src"
        test_dir = tmp_path / "tests"
        modified_file = src_dir / "module.py"
        file_a = src_dir / "a.py"
        file_b = src_dir / "b.py"
        # Create a chain: module.py (foo) -> a.py (bar) -> b.py
        iteration = 0

        def mock_find_references(symbols, search_root):
            nonlocal iteration
            iteration += 1
            if iteration == 1 and "foo" in symbols:
                return {file_a}
            elif iteration == 3 and "bar" in symbols:
                return {file_b}
            return set()

        def mock_get_symbols(file_path):
            if file_path == modified_file:
                return {"foo"}
            elif file_path == file_a:
                return {"bar"}
            elif file_path == file_b:
                return {"baz"}
            return set()

        with (
            patch(
                "tests.conftest_helpers.get_modified_files",
                return_value=[modified_file],
            ),
            patch(
                "tests.conftest_helpers.get_defined_symbols",
                side_effect=mock_get_symbols,
            ),
            patch(
                "tests.conftest_helpers.find_files_referencing_symbols",
                side_effect=mock_find_references,
            ),
        ):
            result = compute_taint(src_dir, test_dir)
            # Chain should propagate through all files
            assert modified_file in result.tainted_files
            assert file_a in result.tainted_files
            assert file_b in result.tainted_files
            # All symbols in the chain should be tainted
            assert "foo" in result.tainted_symbols
            assert "bar" in result.tainted_symbols
            assert "baz" in result.tainted_symbols


class TestIsTestTainted:
    """Tests for checking if individual test files are tainted."""

    def test_tainted_file_returns_true(self):
        """Should return True for files in taint analysis."""
        test_file = Path("/tests/test_foo.py")
        taint = TaintAnalysis(
            tainted_files={test_file},
            tainted_symbols={"foo"},
            initial_modifications=set(),
        )
        assert is_test_tainted(test_file, taint) is True

    def test_untainted_file_returns_false(self):
        """Should return False for files not in taint analysis."""
        test_file = Path("/tests/test_foo.py")
        other_file = Path("/tests/test_bar.py")
        taint = TaintAnalysis(
            tainted_files={other_file},
            tainted_symbols={"bar"},
            initial_modifications=set(),
        )
        assert is_test_tainted(test_file, taint) is False


class TestShouldRunTestsBasedOnTaint:
    """Tests for taint-based test execution decisions."""

    def test_runs_when_test_all_set(self, tmp_path, monkeypatch):
        """Should always run tests when TEST_ALL is set."""
        monkeypatch.setenv("TEST_ALL", "1")
        result = should_run_tests_based_on_taint(tmp_path, tmp_path)
        assert result is True

    def test_skips_when_no_modifications(self, tmp_path, monkeypatch):
        """Should skip tests when no files are modified."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        with patch("tests.conftest_helpers.get_modified_files", return_value=[]):
            result = should_run_tests_based_on_taint(
                tmp_path / "tests", tmp_path / "src"
            )
            assert result is False

    def test_runs_when_test_file_tainted(self, tmp_path, monkeypatch):
        """Should run tests when test files are tainted."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        test_dir = tmp_path / "tests"
        src_dir = tmp_path / "src"
        test_file = test_dir / "test_foo.py"
        src_file = src_dir / "foo.py"
        # Mock compute_taint to return a tainted test file
        mock_taint = TaintAnalysis(
            tainted_files={src_file, test_file},
            tainted_symbols={"foo"},
            initial_modifications={src_file},
        )
        with patch("tests.conftest_helpers.compute_taint", return_value=mock_taint):
            result = should_run_tests_based_on_taint(test_dir, src_dir)
            assert result is True

    def test_skips_when_only_source_files_tainted(self, tmp_path, monkeypatch):
        """Should skip tests when only source files are tainted, not test files."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        test_dir = tmp_path / "tests"
        src_dir = tmp_path / "src"
        src_file = src_dir / "foo.py"
        # Mock compute_taint to return only source file tainted
        mock_taint = TaintAnalysis(
            tainted_files={src_file},
            tainted_symbols={"foo"},
            initial_modifications={src_file},
        )
        with patch("tests.conftest_helpers.compute_taint", return_value=mock_taint):
            result = should_run_tests_based_on_taint(test_dir, src_dir)
            assert result is False


class TestLegacyCompatibility:
    """Tests for backward compatibility with legacy function."""

    def test_dependencies_function_delegates_to_taint(self, tmp_path, monkeypatch):
        """Legacy function should delegate to taint-based implementation."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        with patch(
            "tests.conftest_helpers.should_run_tests_based_on_taint"
        ) as mock_taint:
            mock_taint.return_value = True
            result = should_run_tests_based_on_dependencies(
                tmp_path / "tests", tmp_path / "src"
            )
            assert result is True
            mock_taint.assert_called_once()


class TestShouldRunBaseTests:
    """Tests for base module taint checking with cross-module propagation."""

    def test_runs_when_test_all_set(self, tmp_path, monkeypatch):
        """Should always run tests when TEST_ALL is set."""
        monkeypatch.setenv("TEST_ALL", "1")
        result = should_run_base_tests(tmp_path, tmp_path)
        assert result is True

    def test_skips_when_no_modifications(self, tmp_path, monkeypatch):
        """Should skip tests when no files are modified anywhere."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        with patch("tests.conftest_helpers.compute_taint") as mock_compute:
            mock_compute.return_value = TaintAnalysis(
                tainted_files=set(),
                tainted_symbols=set(),
                initial_modifications=set(),
            )
            result = should_run_base_tests(tmp_path, tmp_path)
            assert result is False

    def test_runs_when_base_source_file_tainted(self, tmp_path, monkeypatch):
        """Should run tests when a base-level source file is tainted."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        test_dir = tmp_path / "tests"
        src_dir = tmp_path / "src"
        base_file = src_dir / "models.py"
        # Create the file so glob finds it
        base_file.parent.mkdir(parents=True, exist_ok=True)
        base_file.touch()
        mock_taint = TaintAnalysis(
            tainted_files={base_file},
            tainted_symbols={"Term"},
            initial_modifications={base_file},
        )
        with patch("tests.conftest_helpers.compute_taint", return_value=mock_taint):
            result = should_run_base_tests(test_dir, src_dir)
            assert result is True

    def test_runs_when_base_test_file_tainted(self, tmp_path, monkeypatch):
        """Should run tests when a base-level test file is tainted."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        test_dir = tmp_path / "tests"
        src_dir = tmp_path / "src"
        test_file = test_dir / "test_models.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.touch()
        mock_taint = TaintAnalysis(
            tainted_files={test_file},
            tainted_symbols={"foo"},
            initial_modifications=set(),
        )
        with patch("tests.conftest_helpers.compute_taint", return_value=mock_taint):
            result = should_run_base_tests(test_dir, src_dir)
            assert result is True

    def test_skips_when_only_submodule_files_tainted(self, tmp_path, monkeypatch):
        """Should skip tests when only submodule files are tainted (not base files)."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        test_dir = tmp_path / "tests"
        src_dir = tmp_path / "src"
        submodule_file = src_dir / "search" / "backends" / "pubmed.py"
        submodule_test = test_dir / "search" / "test_pubmed.py"
        mock_taint = TaintAnalysis(
            tainted_files={submodule_file, submodule_test},
            tainted_symbols={"PubMedBackend"},
            initial_modifications={submodule_file},
        )
        with patch("tests.conftest_helpers.compute_taint", return_value=mock_taint):
            result = should_run_base_tests(test_dir, src_dir)
            assert result is False

    def test_runs_when_submodule_change_taints_base_file(self, tmp_path, monkeypatch):
        """Should run tests when submodule modification taints a base source file."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        test_dir = tmp_path / "tests"
        src_dir = tmp_path / "src"
        submodule_file = src_dir / "search" / "backends" / "pubmed.py"
        base_file = src_dir / "cli.py"
        # Create base_file so glob finds it
        base_file.parent.mkdir(parents=True, exist_ok=True)
        base_file.touch()
        # Taint originates from submodule but reaches base file
        mock_taint = TaintAnalysis(
            tainted_files={submodule_file, base_file},
            tainted_symbols={"PubMedBackend"},
            initial_modifications={submodule_file},
        )
        with patch("tests.conftest_helpers.compute_taint", return_value=mock_taint):
            result = should_run_base_tests(test_dir, src_dir)
            # This is the key behavior: submodule change triggers base tests
            # because taint propagated to a base file
            assert result is True

    def test_computes_taint_from_entire_source_tree(self, tmp_path, monkeypatch):
        """Should compute taint from all modifications, not just base files."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        test_dir = tmp_path / "tests"
        src_dir = tmp_path / "src"
        with patch("tests.conftest_helpers.compute_taint") as mock_compute:
            mock_compute.return_value = TaintAnalysis(
                tainted_files=set(),
                tainted_symbols=set(),
                initial_modifications=set(),
            )
            should_run_base_tests(test_dir, src_dir)
            # Verify compute_taint called with immediate_only=False
            # (to capture submodule modifications)
            mock_compute.assert_called_once_with(
                src_dir, test_dir, immediate_only=False
            )


class TestImmediateOnlyFiltering:
    """Tests for immediate_only parameter (base pseudo-module support)."""

    def test_immediate_only_excludes_subdirectories(self, tmp_path):
        """Should only include immediate children when immediate_only=True."""
        src_dir = tmp_path / "src"
        subdir = src_dir / "submodule"
        root_file = src_dir / "root.py"
        sub_file = subdir / "nested.py"
        # Mock git to return both files
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout=f"{root_file}\n{sub_file}\n", returncode=0
            )
            # With immediate_only=False, both files returned
            result_all = get_modified_files(src_dir, immediate_only=False)
            assert len(result_all) == 2
            # With immediate_only=True, only root file returned
            result_immediate = get_modified_files(src_dir, immediate_only=True)
            assert len(result_immediate) == 1
            assert result_immediate[0].parent == src_dir.resolve()

    def test_compute_taint_respects_immediate_only(self, tmp_path):
        """Taint computation should respect immediate_only parameter."""
        src_dir = tmp_path / "src"
        test_dir = tmp_path / "tests"
        root_file = src_dir / "root.py"
        sub_file = src_dir / "submodule" / "nested.py"

        # Mock get_modified_files to return different results based on directory and immediate_only
        def mock_get_modified(path, immediate_only=False):
            if path == test_dir:
                # No modified test files
                return []
            elif path == src_dir:
                # For source dir, respect immediate_only
                return [root_file] if immediate_only else [root_file, sub_file]
            return []

        with (
            patch(
                "tests.conftest_helpers.get_modified_files",
                side_effect=mock_get_modified,
            ),
            patch("tests.conftest_helpers.get_defined_symbols", return_value={"foo"}),
            patch(
                "tests.conftest_helpers.find_files_referencing_symbols",
                return_value=set(),
            ),
        ):
            # With immediate_only=True, only root file is seed
            taint_immediate = compute_taint(src_dir, test_dir, immediate_only=True)
            assert len(taint_immediate.initial_modifications) == 1
            # With immediate_only=False, both files are seeds
            taint_all = compute_taint(src_dir, test_dir, immediate_only=False)
            assert len(taint_all.initial_modifications) == 2

    def test_should_run_tests_with_immediate_only(self, tmp_path, monkeypatch):
        """Test execution decision should respect immediate_only."""
        monkeypatch.delenv("TEST_ALL", raising=False)
        test_dir = tmp_path / "tests"
        src_dir = tmp_path / "src"
        with patch("tests.conftest_helpers.compute_taint") as mock_compute:
            mock_compute.return_value = TaintAnalysis(
                tainted_files=set(),
                tainted_symbols=set(),
                initial_modifications=set(),
            )
            should_run_tests_based_on_taint(test_dir, src_dir, immediate_only=True)
            # Verify immediate_only was passed through
            mock_compute.assert_called_once_with(src_dir, test_dir, immediate_only=True)


class TestSkipUnlessTainted:
    """Tests for skip_unless_tainted factory function and path resolution."""

    def test_repo_root_calculation_for_base_conftest(self, tmp_path):
        """Repo root should be correctly calculated for tests/conftest.py."""
        # Create minimal project structure
        repo_root = tmp_path
        (repo_root / "src").mkdir()
        (repo_root / "tests").mkdir()
        (repo_root / "src" / "interaction_finder").mkdir()
        conftest = repo_root / "tests" / "conftest.py"
        conftest.write_text("from tests.conftest_helpers import skip_unless_tainted\n")
        # Simulate path calculation in skip_unless_tainted
        test_dir = conftest.parent  # tests/
        calculated_root = test_dir
        while calculated_root.name != "" and not (
            (calculated_root / "tests").exists() and (calculated_root / "src").exists()
        ):
            calculated_root = calculated_root.parent
        # Should find repo root
        assert calculated_root == repo_root
        assert (calculated_root / "src" / "interaction_finder").exists()

    def test_repo_root_calculation_for_submodule_conftest(self, tmp_path):
        """Repo root should be correctly calculated for tests/fetcher/conftest.py."""
        # Create minimal project structure
        repo_root = tmp_path
        (repo_root / "src").mkdir()
        (repo_root / "tests").mkdir()
        (repo_root / "tests" / "fetcher").mkdir()
        (repo_root / "src" / "interaction_finder").mkdir()
        (repo_root / "src" / "interaction_finder" / "fetcher").mkdir()
        conftest = repo_root / "tests" / "fetcher" / "conftest.py"
        conftest.write_text("from tests.conftest_helpers import skip_unless_tainted\n")
        # Simulate path calculation in skip_unless_tainted
        test_dir = conftest.parent  # tests/fetcher/
        calculated_root = test_dir
        while calculated_root.name != "" and not (
            (calculated_root / "tests").exists() and (calculated_root / "src").exists()
        ):
            calculated_root = calculated_root.parent
        # Should find repo root (not tests/)
        assert calculated_root == repo_root
        assert (calculated_root / "src" / "interaction_finder" / "fetcher").exists()

    def test_repo_root_calculation_regression(self, tmp_path):
        """Regression test: subdirectory conftest should not mistake tests/ for repo root.

        This test captures the bug where:
        - tests/fetcher/conftest.py has test_dir = tests/fetcher/
        - test_dir.parent = tests/
        - The OLD code set repo_root = test_dir.parent (tests/)
        - This caused src_root = tests/src/interaction_finder (WRONG!)

        The NEW code walks up until finding a directory with both tests/ and src/.
        """
        # Create structure that would trigger the bug
        repo_root = tmp_path
        (repo_root / "src").mkdir()
        (repo_root / "src" / "interaction_finder").mkdir()
        (repo_root / "tests").mkdir()
        (repo_root / "tests" / "fetcher").mkdir()
        conftest = repo_root / "tests" / "fetcher" / "conftest.py"
        conftest.write_text("")
        # Old buggy logic
        test_dir_old = conftest.parent  # tests/fetcher/
        repo_root_old_buggy = test_dir_old.parent  # tests/ (WRONG!)
        src_root_old_buggy = repo_root_old_buggy / "src" / "interaction_finder"
        # This would be the WRONG path under the bug
        assert not src_root_old_buggy.exists()
        # New correct logic (what skip_unless_tainted does now)
        test_dir_new = conftest.parent
        repo_root_new = test_dir_new
        while repo_root_new.name != "" and not (
            (repo_root_new / "tests").exists() and (repo_root_new / "src").exists()
        ):
            repo_root_new = repo_root_new.parent
        src_root_new = repo_root_new / "src" / "interaction_finder"
        # This should be the CORRECT path
        assert src_root_new.exists()
        assert repo_root_new == repo_root
