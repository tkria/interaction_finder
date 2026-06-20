"""Tests for the persisted user-default config store.

The autouse isolate_platform_dirs fixture redirects user_config_dir into a temp
dir, so these never touch the real ~/.config.
"""

from __future__ import annotations

from interaction_finder.settings import IfetcherConfig
from interaction_finder.web.config_store import (
    clear_default_config,
    load_default_config,
    save_default_config,
)


def test_load_returns_none_when_unset():
    assert load_default_config() is None


def test_save_then_load_round_trips():
    cfg = IfetcherConfig().model_dump()
    cfg["stage"]["search"]["max_rounds"] = 8
    save_default_config(cfg)
    loaded = load_default_config()
    assert loaded is not None
    assert loaded["stage"]["search"]["max_rounds"] == 8


def test_saved_dict_is_a_valid_config():
    cfg = IfetcherConfig().model_dump()
    cfg["stage"]["search"]["max_rounds"] = 8
    save_default_config(cfg)
    # The persisted default must reload into a valid IfetcherConfig.
    reloaded = IfetcherConfig.model_validate(load_default_config())
    assert reloaded.stage.search.max_rounds == 8


def test_clear_removes_the_file():
    save_default_config(IfetcherConfig().model_dump())
    assert load_default_config() is not None
    clear_default_config()
    assert load_default_config() is None


def test_corrupt_file_yields_none(tmp_path, monkeypatch):
    import interaction_finder.web.config_store as store

    bad = tmp_path / "config" / "config.toml"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("this is = = not valid toml", encoding="utf-8")
    assert load_default_config() is None
