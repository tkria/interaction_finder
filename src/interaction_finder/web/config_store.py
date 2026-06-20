"""Persisted user-default configuration for the web UI.

The config editor can save its current values as the user's default config: a
TOML file under ``platformdirs.user_config_dir``. When a run is started without
an explicit ``--config`` and the checkpoint carries no embedded config, this
saved default becomes the baseline (see ``run_manager._baseline_config``).

Stored as TOML to match the project's on-disk config format, so the file is
also hand-editable and interchangeable with a ``config.toml``.
"""

from pathlib import Path

import tomli
import tomli_w
from platformdirs import user_config_dir

_CONFIG_FILE = "config.toml"


def default_config_path() -> Path:
    """Path to the saved default config (its parent dir is created on demand)."""
    config_dir = Path(user_config_dir("interaction-finder"))
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / _CONFIG_FILE


def load_default_config() -> dict | None:
    """Return the saved default config dict, or None if none is saved.

    Returns None on a missing or unreadable file rather than raising, so a
    corrupt default never blocks starting a run from spec defaults.
    """
    path = default_config_path()
    if not path.exists():
        return None
    try:
        return tomli.loads(path.read_text("utf-8"))
    except (tomli.TOMLDecodeError, OSError):
        return None


def save_default_config(config: dict) -> Path:
    """Write ``config`` as the user's default config; return the file path.

    Parameters:
        config: A full config dict (as produced by IfetcherConfig.model_dump).

    None-valued keys are dropped before writing -- TOML has no null, and an
    absent key reloads as the field's None default, so the round-trip is exact.
    """
    path = default_config_path()
    path.write_text(tomli_w.dumps(_drop_none(config)), encoding="utf-8")
    return path


def _drop_none(value):
    """Recursively drop None values from dicts (lists/scalars pass through)."""
    if isinstance(value, dict):
        return {k: _drop_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_none(v) for v in value]
    return value


def clear_default_config() -> None:
    """Remove the saved default config, if present."""
    default_config_path().unlink(missing_ok=True)
