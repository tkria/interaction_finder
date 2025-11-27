"""Tests for version module."""

import pytest

from interaction_finder.checkpoint import PipelineCheckpoint
from interaction_finder.resources import ResourcePool
from interaction_finder.upgrade import create_empty_checkpoint
from interaction_finder.version import (
    check_checkpoint_version,
    format_version_display,
    get_version_string,
    is_breaking_change,
    parse_version_string,
)


class TestGetVersionString:
    """Tests for get_version_string function."""

    def test_returns_string(self):
        """Version string is non-empty."""
        result = get_version_string()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_version_marker(self):
        """Version string contains 'v' marker."""
        assert "v" in get_version_string()

    def test_format_with_git_info(self):
        """With git info, format is <count>v<version>#<hash>."""
        result = get_version_string()
        if "#" in result:  # Has git info
            count, semver, hash_part = parse_version_string(result)
            assert count is not None
            assert len(hash_part) >= 7

    def test_format_without_git_info(self):
        """Without git info, format is v<version>."""
        result = get_version_string()
        if "#" not in result:  # No git info
            assert result.startswith("v")


class TestParseVersionString:
    """Tests for parse_version_string function."""

    @pytest.mark.parametrize(
        "version_str,expected_count,expected_semver,expected_hash",
        [
            # Full format: <count>v<semver>#<hash>
            ("472v0.1.0#3691509", 472, (0, 1, 0), "3691509"),
            ("1v1.0.0#abcdef0", 1, (1, 0, 0), "abcdef0"),
            ("9999v10.20.30#1234567", 9999, (10, 20, 30), "1234567"),
            # No hash: <count>v<semver>
            ("100v1.2.3", 100, (1, 2, 3), None),
            ("1v0.0.1", 1, (0, 0, 1), None),
            # No count (fallback): v<semver>
            ("v0.1.0", None, (0, 1, 0), None),
            ("v1.0.0", None, (1, 0, 0), None),
            ("v10.20.30", None, (10, 20, 30), None),
            # No count with hash: v<semver>#<hash>
            ("v0.1.0#abc1234", None, (0, 1, 0), "abc1234"),
            ("v2.0.0#deadbeef", None, (2, 0, 0), "deadbeef"),
        ],
    )
    def test_parse_all_format_variants(
        self, version_str, expected_count, expected_semver, expected_hash
    ):
        """Parse all supported version format variants."""
        count, semver, hash_part = parse_version_string(version_str)
        assert count == expected_count
        assert semver == expected_semver
        assert hash_part == expected_hash

    def test_invalid_format_raises(self):
        """Invalid version strings raise ValueError."""
        with pytest.raises(ValueError):
            parse_version_string("invalid")
        with pytest.raises(ValueError):
            parse_version_string("1.2.3")  # Missing 'v'


class TestFormatVersionDisplay:
    """Tests for format_version_display function."""

    @pytest.mark.parametrize(
        "version_str,expected",
        [
            ("472v0.1.0#3691509", "0.1.0 (472#3691509)"),
            ("1v1.0.0#abc", "1.0.0 (1#abc)"),
            ("100v1.2.3", "1.2.3 (100)"),
            ("v0.1.0", "0.1.0"),
            ("v2.0.0#deadbeef", "2.0.0 (#deadbeef)"),
        ],
    )
    def test_format_variants(self, version_str, expected):
        """Format all version string variants for display."""
        assert format_version_display(version_str) == expected


class TestIsBreakingChange:
    """Tests for is_breaking_change function."""

    def test_major_bump_is_breaking(self):
        """Major version bump is always breaking."""
        assert is_breaking_change((1, 0, 0), (2, 0, 0)) is True
        assert is_breaking_change((0, 1, 0), (1, 0, 0)) is True

    def test_minor_bump_breaking_for_0x(self):
        """Minor version bump is breaking for 0.x versions."""
        assert is_breaking_change((0, 1, 0), (0, 2, 0)) is True
        assert is_breaking_change((0, 1, 5), (0, 2, 0)) is True

    def test_minor_bump_not_breaking_for_1x(self):
        """Minor version bump is not breaking for 1.x+ versions."""
        assert is_breaking_change((1, 1, 0), (1, 2, 0)) is False
        assert is_breaking_change((2, 0, 0), (2, 5, 0)) is False

    def test_patch_bump_not_breaking(self):
        """Patch version bump is never breaking."""
        assert is_breaking_change((0, 1, 0), (0, 1, 1)) is False
        assert is_breaking_change((1, 0, 0), (1, 0, 1)) is False

    def test_same_version_not_breaking(self):
        """Same version is not breaking."""
        assert is_breaking_change((0, 1, 0), (0, 1, 0)) is False
        assert is_breaking_change((1, 2, 3), (1, 2, 3)) is False


class TestCheckCheckpointVersion:
    """Tests for check_checkpoint_version function."""

    def test_none_version_is_breaking(self):
        """Checkpoint without version is treated as breaking."""
        is_older, is_breaking = check_checkpoint_version(None)
        assert is_older is True
        assert is_breaking is True

    def test_same_version(self):
        """Same version as current is not older or breaking."""
        current = get_version_string()
        is_older, is_breaking = check_checkpoint_version(current)
        assert is_older is False
        assert is_breaking is False

    def test_older_by_commit_count(self):
        """Lower commit count is detected as older."""
        _, semver, _ = parse_version_string(get_version_string())
        semver_str = f"{semver[0]}.{semver[1]}.{semver[2]}"
        older_version = f"1v{semver_str}#0000000"
        is_older, _ = check_checkpoint_version(older_version)
        assert is_older is True

    def test_breaking_minor_bump_0x(self):
        """Minor bump in 0.x is detected as breaking."""
        _, is_breaking = check_checkpoint_version("1v0.0.0#0000000")
        _, cur_semver, _ = parse_version_string(get_version_string())
        if cur_semver[0] == 0:  # Current is 0.x
            assert is_breaking is True


class TestCheckpointVersionIntegration:
    """Tests for version field in PipelineCheckpoint."""

    def test_create_empty_checkpoint_has_version(self):
        """create_empty_checkpoint sets created_by field."""
        checkpoint = create_empty_checkpoint("test topic")
        assert checkpoint.created_by == get_version_string()

    def test_checkpoint_without_version_loads_as_none(self):
        """Old checkpoints without created_by field load with None."""
        old_json = '{"topic": "test", "resources": {"resource_map": {}}}'
        checkpoint = PipelineCheckpoint.model_validate_json(old_json)
        assert checkpoint.created_by is None

    def test_checkpoint_roundtrip_preserves_version(self):
        """Version survives JSON serialization roundtrip."""
        original = create_empty_checkpoint("test topic")
        json_str = original.model_dump_json()
        loaded = PipelineCheckpoint.model_validate_json(json_str)
        assert loaded.created_by == original.created_by
