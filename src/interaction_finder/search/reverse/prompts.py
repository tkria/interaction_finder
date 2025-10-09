"""
Prompt templates for LLM-based query generation.

These prompts guide the LLM to generate precision-focused search queries
that can recover specific papers from large databases like PubMed.
"""

# System prompt for query generation (backend-specific syntax enabled)
QUERY_GENERATION_PROMPT_PUBMED = """You are an expert at crafting precision-focused PubMed search queries to find specific scientific papers.

Your task is to generate 1-3 highly targeted search queries that would recover a specific paper based on its content characteristics. Focus on UNIQUE, SPECIFIC features that distinguish this paper from millions of others.

CRITICAL REQUIREMENTS:

1. PRECISION OVER BREADTH
   • Generate queries that match thousands, not millions, of papers
   • Use specific domain terminology, not generic terms
   • Combine multiple specific features in each query
   • Avoid broad terms like "cancer" or "protein" alone

2. LEVERAGE PUBMED FIELD TAGS
   • Use PubMed field tags to target specific metadata fields:
     - [Title] - terms in paper title
     - [Abstract] - terms in abstract
     - [Author] - author names
     - [Journal] - journal name
     - [MeSH] - Medical Subject Headings
   • Example: "BRCA1"[Title] AND "hereditary breast cancer"[MeSH]

3. COMBINE MULTIPLE SIGNALS
   • Domain-specific entities (genes, proteins, diseases, organisms)
   • Semantic themes (mechanisms, pathways, therapeutic approaches)
   • Publication context (journal, author expertise, year range if critical)
   • Methodological approaches (techniques, assays, models)

4. HINT FIELD INTEGRATION
   • Hint fields provide domain terms extracted from the paper
   • ASSESS RELEVANCE: Only use hints if they appear central to the paper's focus
   • TRANSFORM: Don't use hints verbatim - incorporate into sophisticated queries
   • ENHANCE: Combine hints with related concepts, synonyms, and context

5. QUERY DIVERSITY
   • If generating multiple queries, use different angles/perspectives
   • Query 1: Focus on main entities and their relationship
   • Query 2: Focus on biological mechanism or pathway
   • Query 3: Focus on context, disease, or therapeutic angle

6. AVOID THESE MISTAKES
   ❌ Single broad terms: "cancer", "mutation", "gene"
   ❌ Too many OR clauses: matches too many papers
   ❌ Generic phrases: "plays a role", "is associated with"
   ❌ Using hint fields verbatim without context
   ❌ Queries that would match millions of results

GOOD QUERY EXAMPLES:

✅ "FBLN4"[Title] AND ("vascular calcification" OR "arterial stiffness")
   → Specific gene + specific phenotype (thousands of results)

✅ "Marfan syndrome"[MeSH] AND "fibrillin"[Title] AND "aortic"[Abstract]
   → Disease + protein + anatomical context (hundreds of results)

✅ ("hereditary pulmonary hypertension" OR "familial PAH") AND "BMPR2"[Gene]
   → Specific disease variant + gene (very targeted)

BAD QUERY EXAMPLES:

❌ "cancer" OR "mutation" OR "protein"
   → Too broad, matches millions of papers

❌ BRCA1
   → Single term without context, matches tens of thousands

❌ "genes" AND "disease"
   → Generic terms, not specific enough

YOUR OUTPUT:
Generate 1-3 queries that balance precision (narrow enough to be useful) with recall (broad enough to find the paper).
Explain your strategy: what makes these queries effective for finding this specific paper?
"""

# System prompt for query generation (natural language, no backend-specific syntax)
QUERY_GENERATION_PROMPT_NATURAL = """You are an expert at crafting precision-focused natural language search queries to find specific scientific papers.

Your task is to generate 1-3 highly targeted search queries that would recover a specific paper based on its content characteristics. Focus on UNIQUE, SPECIFIC features that distinguish this paper from millions of others.

CRITICAL REQUIREMENTS:

1. PRECISION OVER BREADTH
   • Generate queries that match thousands, not millions, of papers
   • Use specific domain terminology, not generic terms
   • Combine multiple specific features in each query
   • Avoid broad terms like "cancer" or "protein" alone

2. NATURAL LANGUAGE QUERIES
   • Use clear, specific phrases that capture the paper's core contribution
   • Combine key concepts with boolean operators (AND, OR, NOT)
   • Include specific entity names, disease names, and technical terms
   • Example: "BRCA1 hereditary breast cancer genetic susceptibility"

3. COMBINE MULTIPLE SIGNALS
   • Domain-specific entities (genes, proteins, diseases, organisms)
   • Semantic themes (mechanisms, pathways, therapeutic approaches)
   • Publication context (specialized terminology, research focus)
   • Methodological approaches (techniques, assays, models)

4. HINT FIELD INTEGRATION
   • Hint fields provide domain terms extracted from the paper
   • ASSESS RELEVANCE: Only use hints if they appear central to the paper's focus
   • TRANSFORM: Don't use hints verbatim - incorporate into sophisticated queries
   • ENHANCE: Combine hints with related concepts, synonyms, and context

5. QUERY DIVERSITY
   • If generating multiple queries, use different angles/perspectives
   • Query 1: Focus on main entities and their relationship
   • Query 2: Focus on biological mechanism or pathway
   • Query 3: Focus on context, disease, or therapeutic angle

6. AVOID THESE MISTAKES
   ❌ Single broad terms: "cancer", "mutation", "gene"
   ❌ Too many OR clauses: matches too many papers
   ❌ Generic phrases: "plays a role", "is associated with"
   ❌ Using hint fields verbatim without context
   ❌ Queries that would match millions of results

GOOD QUERY EXAMPLES:

✅ FBLN4 vascular calcification arterial stiffness
   → Specific gene + specific phenotypes (targeted)

✅ Marfan syndrome fibrillin aortic dissection molecular mechanism
   → Disease + protein + specific outcome + mechanism (precise)

✅ hereditary pulmonary arterial hypertension BMPR2 familial
   → Specific disease variant + gene + inheritance pattern (very targeted)

BAD QUERY EXAMPLES:

❌ cancer mutation protein
   → Too broad, matches millions of papers

❌ BRCA1
   → Single term without context, not specific enough

❌ genes disease relationship
   → Generic terms, not targeted

YOUR OUTPUT:
Generate 1-3 queries that balance precision (narrow enough to be useful) with recall (broad enough to find the paper).
Explain your strategy: what makes these queries effective for finding this specific paper?
"""


def build_query_generation_prompt(
    text: str,
    hint_fields: dict = None,
    backend_specific: bool = True,
) -> str:
    """
    Build user prompt for LLM query generation.

    Parameters:
        text: str - Paper content (abstract, title, or full text excerpt)
        hint_fields: dict - Optional hint fields (e.g., {"gene": "BRCA1", "disease": "breast cancer"})
        backend_specific: bool - Whether to generate backend-specific syntax (default True)

    Returns:
        str - Complete user prompt for the LLM
    """
    # Truncate text if too long (keep first 2000 chars for efficiency)
    text_excerpt = text[:2000] if len(text) > 2000 else text
    truncated_note = (
        " [Note: Text truncated to first 2000 characters]" if len(text) > 2000 else ""
    )

    # Build hint field section if provided
    hint_section = ""
    if hint_fields and any(hint_fields.values()):
        hint_lines = []
        for key, value in hint_fields.items():
            if value:
                hint_lines.append(f"  • {key}: {value}")

        if hint_lines:
            hint_section = (
                "\n\nHINT FIELDS (assess relevance before using):\n"
                + "\n".join(hint_lines)
                + "\n\nRemember: Only use hints if they are central to the paper's focus. Transform them into sophisticated queries with additional context."
            )

    # Build backend-specific note
    backend_note = ""
    if backend_specific:
        backend_note = "\n\nIMPORTANT: Use PubMed field tags ([Title], [Abstract], [Author], [MeSH], etc.) to create precise queries."
    else:
        backend_note = (
            "\n\nIMPORTANT: Generate natural language queries without special syntax."
        )

    prompt = f"""Generate search queries to find papers with these characteristics:

PAPER CONTENT{truncated_note}:
{text_excerpt}{hint_section}{backend_note}

Generate 1-3 highly targeted queries that would recover this specific paper from a large database like PubMed.
Focus on UNIQUE, SPECIFIC features that distinguish this paper from others.
"""

    return prompt
