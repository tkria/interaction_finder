"""Tests for load_config's source precedence, especially the fallback base.

These run in a temp cwd so the standard config.toml search locations cannot
pick up a real file from the project root.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from interaction_finder.cli import load_config
from interaction_finder.settings import IfetcherConfig


@pytest.fixture
def clean_cwd(tmp_path, monkeypatch):
    """Run in an empty dir so no config.toml is auto-discovered."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_no_sources_yields_spec_defaults(clean_cwd):
    config = load_config()
    assert config.stage.search.max_rounds == IfetcherConfig().stage.search.max_rounds


def test_fallback_config_is_used_as_base(clean_cwd):
    base = IfetcherConfig().model_dump()
    base["stage"]["search"]["max_rounds"] = 9
    config = load_config(fallback_config=base)
    assert config.stage.search.max_rounds == 9


def test_overrides_apply_on_top_of_fallback(clean_cwd):
    base = IfetcherConfig().model_dump()
    base["stage"]["search"]["max_rounds"] = 9
    config = load_config(overrides=["stage.search.max_rounds=4"], fallback_config=base)
    assert config.stage.search.max_rounds == 4


def test_config_file_takes_precedence_over_fallback(clean_cwd):
    cfg_file = clean_cwd / "config.toml"
    cfg_file.write_text("[stage.search]\nmax_rounds = 7\n", encoding="utf-8")
    base = IfetcherConfig().model_dump()
    base["stage"]["search"]["max_rounds"] = 9
    config = load_config(fallback_config=base)
    assert config.stage.search.max_rounds == 7  # file wins over fallback


def test_fallback_is_not_mutated(clean_cwd):
    base = IfetcherConfig().model_dump()
    base["stage"]["search"]["max_rounds"] = 9
    load_config(overrides=["stage.search.max_rounds=4"], fallback_config=base)
    assert base["stage"]["search"]["max_rounds"] == 9  # deepcopy protects caller
