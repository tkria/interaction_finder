"""Version information for interaction-finder.

Provides version string in format: <commit_count>v<version>#<short_hash>
Falls back to v<version> if git info unavailable.
"""

import re
import subprocess
from functools import lru_cache
from importlib.metadata import version as get_package_version

# Regex to parse version strings: optional count, required semver, optional hash
_VERSION_RE = re.compile(r"^(\d+)?v(\d+\.\d+\.\d+)(?:#(.+))?$")


def _get_git_info() -> tuple[int, str] | None:
    """Get git commit count and short hash, or None if unavailable."""
    try:
        count = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        hash_ = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return int(count), hash_
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        return None


@lru_cache(maxsize=1)
def get_version_string() -> str:
    """Get version string: '<count>v<version>#<hash>' or 'v<version>' without git."""
    pkg_version = get_package_version("interaction-finder")
    git_info = _get_git_info()
    if git_info is None:
        return f"v{pkg_version}"
    return f"{git_info[0]}v{pkg_version}#{git_info[1]}"


def parse_version_string(
    version_str: str,
) -> tuple[int | None, tuple[int, int, int], str | None]:
    """Parse version string into (commit_count, (major, minor, patch), hash)."""
    match = _VERSION_RE.match(version_str)
    if not match:
        raise ValueError(f"Invalid version string: {version_str}")
    count_str, semver, hash_ = match.groups()
    count = int(count_str) if count_str else None
    major, minor, patch = map(int, semver.split("."))
    return count, (major, minor, patch), hash_


def is_breaking_change(
    old_semver: tuple[int, int, int], new_semver: tuple[int, int, int]
) -> bool:
    """Check if version change is breaking. For 0.x, minor bump is breaking."""
    old_major, old_minor, _ = old_semver
    new_major, new_minor, _ = new_semver
    if new_major != old_major:
        return True
    if old_major == 0 and new_minor != old_minor:
        return True
    return False


def format_version_display(version_str: str) -> str:
    """Format version string for human display: '0.1.0 (472#3691509)' or '0.1.0'."""
    count, semver, hash_ = parse_version_string(version_str)
    semver_str = f"{semver[0]}.{semver[1]}.{semver[2]}"
    if count is not None and hash_ is not None:
        return f"{semver_str} ({count}#{hash_})"
    elif count is not None:
        return f"{semver_str} ({count})"
    elif hash_ is not None:
        return f"{semver_str} (#{hash_})"
    return semver_str


def format_version_tooltip(version_str: str) -> tuple[str, str | None]:
    """Format version for tooltip display.

    Returns:
        (display_text, tooltip_text) where tooltip may be None if no extra info

    Example:
        '474v0.1.0#1822414' -> ('0.1.0', 'rev.474 #1822414')
        'v0.1.0' -> ('0.1.0', None)
    """
    count, semver, hash_ = parse_version_string(version_str)
    semver_str = f"{semver[0]}.{semver[1]}.{semver[2]}"
    # Build tooltip only if we have extra info
    if count is not None and hash_ is not None:
        tooltip = f"rev.{count} #{hash_}"
        return semver_str, tooltip
    elif count is not None:
        return semver_str, f"rev.{count}"
    elif hash_ is not None:
        return semver_str, f"#{hash_}"
    return semver_str, None


def check_checkpoint_version(checkpoint_version: str | None) -> tuple[bool, bool]:
    """Check checkpoint version against current.

    Returns:
        (is_older, is_breaking) tuple
    """
    if checkpoint_version is None:
        return True, True  # Unknown version treated as breaking
    cp_count, cp_semver, _ = parse_version_string(checkpoint_version)
    cur_count, cur_semver, _ = parse_version_string(get_version_string())
    # Determine if older
    if cp_count is not None and cur_count is not None:
        is_older = cp_count < cur_count
    else:
        is_older = cp_semver < cur_semver
    return is_older, is_breaking_change(cp_semver, cur_semver)
