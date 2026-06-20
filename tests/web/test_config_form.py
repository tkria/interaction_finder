"""Tests for the spec-generated config form HTML."""

from __future__ import annotations

import re

from interaction_finder.settings import IfetcherConfig
from interaction_finder.web.config_form import build_config_form_html


def _html(overrides=None):
    cfg = IfetcherConfig().model_dump()
    return build_config_form_html(cfg if overrides is None else overrides)


def test_top_sections_are_open_accordions():
    html = _html()
    # tools, stage, output -> three top-level open <details>.
    # agents, tools, stage, output -> four top-level open <details>.
    assert html.count("cfg-top") == 4


def test_known_field_has_path_bounds_default_and_value():
    html = _html()
    assert 'name="stage.search.max_rounds"' in html
    assert 'min="1"' in html and 'max="15"' in html
    assert 'data-default="3"' in html


def test_bool_field_renders_as_switch():
    html = _html()
    assert 'type="checkbox" role="switch" name="stage.search.enabled"' in html
    assert 'data-default="true"' in html


def test_value_reflects_effective_config_not_just_default():
    cfg = IfetcherConfig().model_dump()
    cfg["stage"]["search"]["max_rounds"] = 9
    html = build_config_form_html(cfg)
    # The pre-filled value is 9 while the spec default stays 3.
    m = re.search(r'name="stage\.search\.max_rounds"[^>]*', html)
    assert m and 'value="9"' in m.group(0) and 'data-default="3"' in m.group(0)


def test_collection_and_dict_fields_are_skipped():
    html = _html()
    # sources is a list; modes is an arbitrary-preset dict -- neither appears.
    assert "perplexica.sources" not in html
    assert 'name="modes' not in html
    # model_settings (a free-form dict on AgentSpec) has no clean control.
    assert ".model_settings" not in html


def test_nested_models_become_subaccordions():
    html = _html()
    # tools.search.perplexica is a nested model -> its fields are namespaced.
    assert 'name="tools.search.perplexica.base_url"' in html


def test_agents_hierarchy_is_rendered():
    html = _html()
    # Global, module-default, and per-agent paths all appear.
    assert 'name="agents._.llm"' in html
    assert 'name="agents.extraction._.llm"' in html
    assert 'name="agents.extraction.pair_judge.llm"' in html
    # Advanced AgentSpec fields are present (behind the per-agent toggle).
    assert 'name="agents.extraction.pair_judge.retries"' in html


def test_saved_agent_values_prefill():
    cfg = IfetcherConfig().model_dump()
    cfg["agents"] = {"extraction": {"pair_judge": {"llm": "openai:gpt-4o"}}}
    html = build_config_form_html(cfg)
    m = re.search(r'name="agents\.extraction\.pair_judge\.llm"[^>]*', html)
    assert m and 'value="openai:gpt-4o"' in m.group(0)
