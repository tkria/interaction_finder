"""Generate the config editor's form HTML from the IfetcherConfig spec.

The editor is built once from the Pydantic model -- not per request and not in
the client -- so adding or changing a config field updates the form for free.
The walker emits nested ``<details>`` accordions (one per nested model) whose
leaves are Pico form controls named by their dotted config path (e.g.
``stage.search.max_rounds``), pre-filled with the effective value and tagged
with ``data-default`` (the spec default) so the client can highlight and reset
non-default entries.

Dict/list-typed fields (the free-form ``agents`` map, ``modes``, and any
list/tuple field) have no clean scalar control and are skipped; ``log()`` is
not used here -- the skip set is fixed and documented rather than dynamic.

Public API: ``build_config_form_html(effective)``.
"""

from __future__ import annotations

import html
import types
import typing
from typing import Any

from annotated_types import Ge, Le
from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from interaction_finder.settings import IfetcherConfig

# Top-level fields with no clean scalar form (free-form dicts); skipped wholesale.
# `agents` is also a dict but rendered specially (its known hierarchy), so it is
# NOT skipped -- only `modes` (arbitrary user presets) is.
_SKIP_SECTIONS = frozenset({"modes"})

# The agents hierarchy is keyed (not a fixed model), so the known module/agent
# names are enumerated here. Each level resolves an AgentSpec, with fallback
# agents.<module>.<agent> -> agents.<module>._ -> agents._ -> code default.
_AGENT_NAMES: dict[str, tuple[str, ...]] = {
    "keywords": (
        "query_expander",
        "result_selector",
        "keyword_evaluator",
        "document_summarizer",
        "reflector",
    ),
    "search": ("goal_planner", "query_generator", "result_selector", "reflector"),
    "extraction": (
        "entity",
        "entity_merger",
        "proximal_pair",
        "pair_judge",
        "cross_judge",
    ),
}
# AgentSpec's primary field (always shown) and the advanced ones (behind a
# per-agent toggle). model_settings (a free-form dict) has no clean control.
_AGENT_PRIMARY = "llm"
_AGENT_ADVANCED = ("expertise", "instruction", "retries", "instrument", "system_prompt")


def build_config_form_html(effective: dict[str, Any]) -> str:
    """Return the config editor's inner HTML for the given effective config.

    Parameters:
        effective: The config dict whose values pre-fill the form (typically
            ``IfetcherConfig().model_dump()`` overlaid with any saved default).

    Returns the accordion markup for every renderable top-level section; the
    caller wraps it in the modal shell.
    """
    sections = []
    for name, field in IfetcherConfig.model_fields.items():
        if name in _SKIP_SECTIONS:
            continue
        if name == "agents":
            sections.append(
                _accordion(
                    "agents", _render_agents(effective.get("agents", {})), top=True
                )
            )
            continue
        model = _model_type(field.annotation)
        if model is None:
            continue  # top level holds only nested models we care about
        body = _render_model(model, name, effective.get(name, {}))
        sections.append(_accordion(name, body, top=True))
    return "".join(sections)


# === helpers: type inspection ===
def _unwrap_optional(ann: Any) -> Any:
    """Return X for ``X | None`` (and plain X otherwise)."""
    if typing.get_origin(ann) in (types.UnionType, typing.Union):
        non_none = [a for a in typing.get_args(ann) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return ann


def _model_type(ann: Any) -> type[BaseModel] | None:
    """Return the BaseModel subclass an annotation resolves to, else None."""
    inner = _unwrap_optional(ann)
    if isinstance(inner, type) and issubclass(inner, BaseModel):
        return inner
    return None


def _scalar_input_type(ann: Any) -> str | None:
    """Map a leaf annotation to a control kind: 'bool' | 'number' | 'text'.

    Returns None for collections/other types that get no control.
    """
    inner = _unwrap_optional(ann)
    if inner is bool:
        return "bool"
    if inner in (int, float):
        return "number"
    if inner is str:
        return "text"
    return None  # list/tuple/dict/unknown -> skipped


def _bounds(field: Any) -> tuple[float | None, float | None]:
    """Extract (min, max) from a field's Ge/Le metadata, if any."""
    lo = hi = None
    for m in field.metadata:
        if isinstance(m, Ge):
            lo = m.ge
        elif isinstance(m, Le):
            hi = m.le
    return lo, hi


# === helpers: agents hierarchy ===
def _render_agents(values: dict) -> str:
    """Render the agents hierarchy: global, then per-module with per-agent rows.

    ``values`` is the (possibly sparse) ``agents`` dict from the effective
    config; missing levels render empty controls (which inherit at runtime).
    """
    global_spec = _render_agent_spec("agents._", _at(values, "_"), "Global default")
    modules = []
    for module, agents in _AGENT_NAMES.items():
        module_vals = _at(values, module)
        body = _render_agent_spec(
            f"agents.{module}._", _at(module_vals, "_"), f"{module} default"
        )
        for agent in agents:
            body += _render_agent_spec(
                f"agents.{module}.{agent}", _at(module_vals, agent), agent
            )
        modules.append(_accordion(module, body, top=False))
    return f'<div class="cfg-body">{global_spec}</div>' + "".join(modules)


def _render_agent_spec(prefix: str, values: dict, label: str) -> str:
    """One agent: its llm field, plus an advanced toggle for the rest.

    ``prefix`` is the dotted path to this AgentSpec (e.g. agents.extraction.entity);
    each field is named ``<prefix>.<field>``.
    """
    spec = IfetcherConfig.AgentSpec
    # The agent name IS the llm field's label: one <label> wraps the name and
    # the input (e.g. "Global default" / "pair judge"), so there is no separate
    # stacked "llm" caption.
    llm_value = values.get("llm") if isinstance(values, dict) else None
    val_attr = "" if not llm_value else f' value="{html.escape(str(llm_value))}"'
    title = html.escape(label.replace("_", " "))
    llm = (
        f'<label class="agent-head"><span class="agent-name">{title}</span>'
        f'<input type="text" name="{prefix}.llm" class="agent-llm" '
        f'placeholder="(inherit)" autocomplete="off" data-default=""{val_attr}></label>'
    )
    advanced = "".join(
        _spec_field(spec, name, prefix, values) for name in _AGENT_ADVANCED
    )
    return (
        f'<div class="agent-row">{llm}'
        f'<details class="agent-advanced"><summary>Advanced</summary>'
        f'<div class="cfg-body">{advanced}</div></details></div>'
    )


def _spec_field(spec: type[BaseModel], name: str, prefix: str, values: dict) -> str:
    """Render one AgentSpec field as a control named ``<prefix>.<name>``."""
    field = spec.model_fields[name]
    kind = _scalar_input_type(field.annotation)
    if kind is None:
        return ""
    value = values.get(name) if isinstance(values, dict) else None
    return _field(f"{prefix}.{name}", name, field, kind, value)


def _at(values: Any, key: str) -> dict:
    """Return values[key] when it's a dict, else an empty dict (sparse-safe)."""
    inner = values.get(key) if isinstance(values, dict) else None
    return inner if isinstance(inner, dict) else {}


# === helpers: rendering ===
def _render_model(model: type[BaseModel], prefix: str, values: dict) -> str:
    """Render a model's fields: scalar leaves first, then nested sub-accordions."""
    leaves, subsections = [], []
    for name, field in model.model_fields.items():
        path = f"{prefix}.{name}"
        value = values.get(name) if isinstance(values, dict) else None
        nested = _model_type(field.annotation)
        if nested is not None:
            body = _render_model(nested, path, value or {})
            subsections.append(_accordion(name, body, top=False))
            continue
        kind = _scalar_input_type(field.annotation)
        if kind is None:
            continue  # collection/unsupported -- skipped by design
        leaves.append(_field(path, name, field, kind, value))
    return "".join(leaves) + "".join(subsections)


def _field(
    path: str, name: str, field: Any, kind: str, value: Any, placeholder: str = ""
) -> str:
    """Render one labelled form control with its description and default."""
    default = field.default
    if default is PydanticUndefined:
        default = None
    label = html.escape(name.replace("_", " "))
    desc = html.escape(field.description or "")
    desc_html = f'<small class="cfg-desc">{desc}</small>' if desc else ""
    ph_attr = f' placeholder="{html.escape(placeholder)}"' if placeholder else ""
    if kind == "bool":
        checked = " checked" if _coerce_bool(value, default) else ""
        control = (
            f'<input type="checkbox" role="switch" name="{path}" '
            f'data-default="{_bool_str(default)}"{checked}>'
        )
        return (
            f'<label class="cfg-field cfg-bool">{control} '
            f"<span>{label}</span>{desc_html}</label>"
        )
    attrs = ""
    if kind == "number":
        lo, hi = _bounds(field)
        if lo is not None:
            attrs += f' min="{_num(lo)}"'
        if hi is not None:
            attrs += f' max="{_num(hi)}"'
        # Integers step by 1; floats allow fractional input.
        attrs += ' step="any"' if _unwrap_optional(field.annotation) is float else ""
    shown = value if value is not None else default
    val_attr = "" if shown is None else f' value="{html.escape(str(shown))}"'
    input_type = "number" if kind == "number" else "text"
    control = (
        f'<input type="{input_type}" name="{path}"{attrs}{ph_attr}{val_attr} '
        f'data-default="{html.escape("" if default is None else str(default))}">'
    )
    return f'<label class="cfg-field">{label}{desc_html}{control}</label>'


# Icons for the top-level config sections (symbol ids declared in ui.html).
_SECTION_ICONS = {
    "agents": "icon-robot",
    "tools": "icon-tools",
    "stage": "icon-stages",
    "output": "icon-output",
}


def _accordion(name: str, body: str, top: bool) -> str:
    """A <details> section.

    A top-level section starts open only when it contains sub-accordions, so
    the structure is visible at a glance; a flat top-level section (just
    fields, e.g. output) and every nested section start collapsed.
    """
    cls = "cfg-section cfg-top" if top else "cfg-section"
    open_attr = " open" if top and "cfg-section" in body else ""
    title = html.escape(name.replace("_", " "))
    icon = ""
    if top and name in _SECTION_ICONS:
        icon = (
            f'<svg class="ui-icon cfg-section-icon" aria-hidden="true" '
            f'focusable="false"><use href="#{_SECTION_ICONS[name]}"/></svg>'
        )
    return (
        f'<details class="{cls}"{open_attr}>'
        f'<summary>{icon}{title}<span class="cfg-count" hidden></span></summary>'
        f'<div class="cfg-body">{body}</div></details>'
    )


def _coerce_bool(value: Any, default: Any) -> bool:
    return bool(value if value is not None else default)


def _bool_str(value: Any) -> str:
    return "true" if value else "false"


def _num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)
