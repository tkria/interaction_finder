"""Subject-trust judge agent (gate_review, strict subject-vs-candidate).

Decides whether a candidate denotes the topic's SUBJECT -- the same entity, or a
taxonomic form/category of it -- versus a distinct entity merely related to it.
Judged PURELY from the taxonomic relation; never from which genes, markers, or
other partners either has. Leakage-clean by construction.

This is the prompt validated in the topic-gate signal experiments over all 60
casestudy topics (gpt-5-mini): pooled ~91% gold-recovery retained while ~48% of
off-subject candidates are rejected. The subject is passed EXPLICITLY (the
topic's anchor entity), not inferred from the topic string -- that removes the
answer-set confusion that broke ligand-receptor topics ("receptors that bind to
X" made the judge reject X itself). See
``casestudies/topic_gate/signal_experiments/scripts/test_strict_gate.py``.
"""

from pydantic_ai.settings import ModelSettings

from interaction_finder.agent_config import agent_getter
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import GateReviewVerdict


get_subject_trust_agent = agent_getter(
    "extraction",
    "subject_trust",
    GateReviewVerdict,
    Deps,
    """You are vetting a TRUSTED answer set for a research topic. You are given a
SUBJECT and a CANDIDATE, both of the same KIND. Decide whether the candidate
denotes the subject -- the same entity, or a taxonomic form or category of it --
as opposed to a distinct entity that is merely related to it. Judge ONLY the
taxonomic relationship between the two; never reason about what either is
associated with, produces, expresses, binds, or is marked by.

ADMIT only when the candidate is one of:
  - the subject itself -- an alias, acronym, synonym, or the same entity under a
    different name, stage, or form;
  - a more specific kind of the subject -- a subtype, form, variant, or named
    member that falls under it;
  - a broader or more general category of which the subject is a significant
    kind, unless that category is so broad the subject is only one of many
    unrelated members;
  - a grouped label that explicitly names the subject as one of its components.

Otherwise REJECT. Relatedness is not identity: reject a candidate that is merely
associated or co-occurring with the subject, that the subject produces or gives
rise to, that is a downstream consequence or product of it, or that is a
separate sibling or relative not itself falling under the subject. Default to
REJECT; when genuinely unsure, REJECT.

Answer with a one-sentence reason naming the relationship, then the boolean.""",
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)
