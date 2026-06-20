"""Shared fixtures for web tests.

Critically, isolates the report cache and recent-history registry from the
user's real platform directories: without this, loading checkpoints, recording
recents, or generating reports during tests would read from and write to the
actual ``platformdirs`` cache/data locations.
"""

import pytest

import interaction_finder.web.config_store as config_store_mod
import interaction_finder.web.recent as recent_mod
import interaction_finder.web.report_cache as report_cache_mod


@pytest.fixture(autouse=True)
def isolate_platform_dirs(tmp_path, monkeypatch):
    """Redirect the report cache, recent registry and config store to a temp dir.

    Patches the ``user_cache_dir`` / ``user_data_dir`` / ``user_config_dir``
    names as imported by the web modules, so every path the modules build lands
    under ``tmp_path`` for the duration of each test. The real ~/.cache,
    ~/.local/share and ~/.config stay untouched.
    """
    cache_dir = tmp_path / "cache"
    data_dir = tmp_path / "data"
    config_dir = tmp_path / "config"
    monkeypatch.setattr(report_cache_mod, "user_cache_dir", lambda _app: str(cache_dir))
    monkeypatch.setattr(recent_mod, "user_data_dir", lambda _app: str(data_dir))
    monkeypatch.setattr(
        config_store_mod, "user_config_dir", lambda _app: str(config_dir)
    )
