"""Document analysis agent.

Combined paper quality assessment and entity extraction in a single pass.
Quality assessment comes first to establish paper understanding before extraction.
"""

from pydantic_ai.settings import ModelSettings

from interaction_finder.agent_config import agent_getter
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.models import DocumentAnalysisOut

_ENTITY_EXTRACTION_RULES = """
**Extraction rules:**
1. Extract only entities of the requested types (gene, disease, protein, etc.)
2. Use canonical names (e.g., "BRCA1" not "BRCA-1")
3. Include all verbatim names as they appear in the text
4. Provide exact quotes supporting each entity
5. Only extract entities clearly relevant to the topic
6. Require clear textual support for every entity
7. Explain your reasoning for each entity extraction

**Alias handling:**
- Aliases must be alternative names for the SAME entity
- Include acronyms, abbreviations, and lexical variants
- If a name includes the base entity PLUS additional qualifiers that significantly
  narrow or change the meaning, extract it as a separate entity instead

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
