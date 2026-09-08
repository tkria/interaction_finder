"""Subject resolver (topic_policy): topic + entity kind -> the named subject.

Identifies which entity, if any, a research topic explicitly names as its
subject of interest, for one entity kind. Run once per target entity kind; the
kind whose result names an entity is the subject/anchor side, and that entity is
the anchor the subject-trust gate judges candidates against.

This is Prompt A of the topic-gate signal experiments (``topic_policy``),
ported verbatim. The gate consumes only the anchor name; the scope booleans and
aliases are carried for the fuller policy but not yet read downstream. The
literal-mention discipline is the load-bearing part: the topic's *named* subject
is returned, never a pre-filled famous-in-the-field set, and a kind the topic
does not name returns empty (so "receptors that bind to X" resolves X as the
ligand subject and leaves the receptor side open for discovery).
"""

from pydantic_ai.settings import ModelSettings

from interaction_finder.agent_config import AGENT_CALL_ERRORS, agent_getter
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import TopicPolicy


get_subject_resolver_agent = agent_getter(
    "extraction",
    "subject_resolver",
    TopicPolicy,
    Deps,
    """For one entity kind, identify which specific entities (if any) the
research topic explicitly names as its subject of interest. The result
drives a filter on extracted entities of this kind.

## The literal-mention test

Only return entities that the topic itself names or refers to (by
synonym, abbreviation, or standard alias) AND that are themselves
entities of the kind currently being resolved. If the topic names an
entity of a DIFFERENT kind, return EMPTY for this kind. (E.g. for
topic "cell-markers for hepatocytes" with kind=cellmarker:
"hepatocyte" is a celltype, not a cellmarker — return empty.
The literal mention must match the resolved kind.)

If the topic does not name any specific entity of this kind, return an
EMPTY `entities` list — do NOT invent a constrained set based on
what's *famous* in the field. The user's discovery question must not
be filtered down to a pre-decided list.

## Per-entity reasoning

For each named entity, write ONE short sentence saying why it's the
topic subject (e.g. "Topic explicitly names this"; "Standard synonym
for the topic disease"). This is informational only and is NOT used
for scope decisions. Do NOT articulate exclusions here.

## Aliases

List strictly equivalent labels: alternative spellings (US/UK),
abbreviations, eponyms, and pragmatic synonyms that the literature
routinely uses interchangeably for the SAME diagnosable thing.

Include pragmatic synonyms even when they're not perfectly equivalent
in strict taxonomy — if the literature commonly conflates them, list
them. (E.g. for "HNPCC" include "Lynch syndrome" and "hereditary
nonpolyposis colorectal cancer"; for "Li-Fraumeni syndrome" include
"LFS".)

Do NOT include broader categories, narrower subtypes, or sibling
disorders — those belong in the scope booleans, not here.

## Scope booleans

`subtypes_in_scope`: usually True when entities is non-empty. False
only for unusually narrow topics ("wild-type p53 specifically").

`supertypes_in_scope`: True when a generally-recognised parent
category, substituted into the topic, would ask substantively the
same research question. False when substitution makes the question
broader-scope (covering many sibling categories the topic didn't
ask about).

`associated_in_scope`: True when the research question naturally
asks about non-taxonomic links — causes, mechanisms, prodromal
signs, characteristic features, contributing factors. Most
"genes/factors associated with X" topics are True (the framing
itself invites associations). Most "markers for X" or "subtypes
of X" topics are False (the framing asks for identity-of-X, not
adjacent biology).

## Examples

Topic: "cell-markers for hepatocytes", kind=celltype
→ entities=[{name: "Hepatocyte",
            reasoning: "Topic explicitly names this cell type.",
            aliases: ["hepatocyte", "Hepatic cell", "hepatic cells", "hepatocytes"]}]
  subtypes_in_scope=true
  supertypes_in_scope=false
  associated_in_scope=false

Topic: "cell-markers for hepatocytes", kind=cellmarker
→ entities=[]
  subtypes_in_scope=false, supertypes_in_scope=false, associated_in_scope=false

Topic: "genes associated with Li-Fraumeni syndrome", kind=gene
→ entities=[]
  WRONG: returning [TP53, CHEK2, ...] because those are known
  Li-Fraumeni genes. The topic does not name them. Do NOT pre-filter.

Topic: "genes associated with Li-Fraumeni syndrome", kind=phenotype
→ entities=[{name: "Li-Fraumeni syndrome",
            reasoning: "Topic explicitly names this phenotype.",
            aliases: ["LFS"]}]
  subtypes_in_scope=true
  supertypes_in_scope=false
  associated_in_scope=true

Topic: "genes associated with HNPCC", kind=phenotype
→ entities=[{name: "HNPCC",
            reasoning: "Topic explicitly names this phenotype.",
            aliases: ["Lynch syndrome", "hereditary nonpolyposis colorectal cancer"]}]
  subtypes_in_scope=true
  supertypes_in_scope=false
  associated_in_scope=true

Topic: "genes associated with cleft palate", kind=phenotype
→ entities=[{name: "Cleft palate",
            reasoning: "Topic explicitly names this phenotype.",
            aliases: ["palatal cleft"]}]
  subtypes_in_scope=true
  supertypes_in_scope=true
  associated_in_scope=true

Topic: "receptors that bind to EGF", kind=ligand
→ entities=[{name: "EGF",
            reasoning: "Topic explicitly names this ligand.",
            aliases: ["Epidermal Growth Factor", "epidermal growth factor"]}]
  subtypes_in_scope=true
  supertypes_in_scope=false
  associated_in_scope=false

Topic: "receptors that bind to EGF", kind=receptor
→ entities=[]
  Even though EGFR is the well-known EGF receptor, the topic asks
  the discovery question 'which receptors?' — do NOT pre-fill.""",
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)


async def resolve_subject(
    topic: str, entity_types: list[str], deps: Deps
) -> tuple[str | None, str | None]:
    """Resolve a topic's subject kind and anchor entity for the trust gate.

    Runs the subject resolver once per distinct entity kind. The kind whose
    result names exactly one entity is the subject/anchor side, and that entity
    is the anchor. Returns ``(None, None)`` -- disabling the gate -- when no kind
    resolves to a single named subject, which is the correct behaviour for a
    genuinely open discovery topic on both sides.

    Parameters:
        topic: The research topic string.
        entity_types: Target entity kinds for the run (duplicates ignored).
        deps: Extraction deps (config + agent semaphore).

    Returns:
        (subject_kind, subject_anchor) -- both None if unresolved.
    """
    agent = get_subject_resolver_agent(deps.config)
    resolved: list[tuple[str, str]] = []
    for kind in dict.fromkeys(entity_types):
        try:
            async with deps.agent_semaphore:
                result = await agent.run(f"Topic: {topic}\nKind: {kind}", deps=deps)
        except AGENT_CALL_ERRORS as e:
            deps.logger.error(
                f"Subject resolution failed for kind {kind!r}: {type(e).__name__}: {e}"
            )
            continue
        names = [e.name for e in result.output.entities if e.name.strip()]
        # Single named subject only: multiple names are ambiguous (the validated
        # backfill discarded them likewise), zero means this kind is open.
        if len(names) == 1:
            resolved.append((kind, names[0]))
    if len(resolved) != 1:
        if len(resolved) > 1:
            deps.logger.info(
                "resolve_subject: %d kinds named a subject (%s); leaving the "
                "gate disabled to avoid an ambiguous anchor.",
                len(resolved),
                ", ".join(k for k, _ in resolved),
            )
        return None, None
    kind, anchor = resolved[0]
    deps.logger.info(f"resolve_subject: subject_kind={kind!r} anchor={anchor!r}")
    return kind, anchor
