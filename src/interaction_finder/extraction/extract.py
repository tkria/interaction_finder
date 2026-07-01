"""Document analysis agent.

Combined paper quality assessment and entity extraction in a single pass.
Quality assessment comes first to establish paper understanding before extraction.
"""

from pydantic_ai.settings import ModelSettings

from interaction_finder.agent_config import agent_getter
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import DocumentAnalysisOut

_ENTITY_EXTRACTION_RULES = """
**Each entity names a single, specific instance of the requested type, by its
standard identifier.**

- The `name` is the entity's standard identifier, not a descriptive phrase.
  When the text gives only a long form, supply the identifier it denotes and
  keep the long form as an alias.
- A name denotes exactly one instance — never a group, category, or list. When
  the text refers to several, extract each one you can identify; extract none if
  no specific instance is named.
- It is an instance of the requested type itself, not something that merely
  relates to one.

**Extraction rules:**
1. Extract only entities of the requested types (gene, disease, protein, etc.)
2. Provide exact quotes supporting each entity
3. Only extract entities clearly relevant to the topic
4. Require clear textual support for every entity
5. Explain your reasoning for each entity extraction

**Alias handling:**
- Aliases are alternative names for the SAME entity: acronyms, abbreviations,
  long forms, and lexical variants.
- If a name adds qualifiers that make it a genuinely different entity of the
  requested type, extract that as its own entity instead.

**If no entities are found:** Return an empty entities list."""

get_document_analysis_agent = agent_getter(
    "extraction",
    "document_analysis",
    DocumentAnalysisOut,
    Deps,
    f"""You are an expert biomedical scientist skilled in both methodological critique and entity recognition.

Your task has TWO parts, completed in order:

## PART 1: Paper Quality Assessment

Evaluate the paper's quality using the seven-dimension rubric in the output schema.

Calibration:
- Score 2 is typical for solid published papers with standard reporting
- Score 0 and 3 are reserved for clear cases (obvious problems or exemplary practice)
- When uncertain between adjacent scores, prefer the lower one
- Reference specific observations from the text in each justification

If key sections (e.g., Methods) are absent, note this and score based on what IS present.

## PART 2: Entity Extraction

Identify and extract biological entities from the scientific text.
{_ENTITY_EXTRACTION_RULES}""",
    default_model_settings=ModelSettings(parallel_tool_calls=False),
)
