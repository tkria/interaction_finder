"""
Prompt templates for LLM-based query generation.

These prompts guide the LLM to generate precision-focused search queries
that can recover specific papers from large databases like PubMed.

This module contains two types of prompts:
1. QUERY_GENERATION_PROMPT_* - For end-to-end LLM query generation (Stage 1+2 combined)
2. QUERY_CONSTRUCTION_PROMPT - For Stage 2 query construction using extracted keywords
"""

# ==============================================================================
# Query Construction Prompts (Stage 2 of two-stage pipeline)
# ==============================================================================

# Base prompt shared by both keyword and content-only modes
QUERY_CONSTRUCTION_BASE = """You are an expert at constructing precision-focused search queries for scientific literature search.

Your task is to build a single, optimized search query based on the information provided.

BACKEND-SPECIFIC SYNTAX:
• For PubMed/structured backends:
  - Use field tags: [Title], [Abstract], [MeSH], [Author], [Gene]
  - Combine terms with AND/OR operators for precision
  - Use quotes for exact phrases when appropriate
• For natural language backends (perplexica, openai):
  - Use conversational phrases or questions
  - Do NOT use Boolean operators (AND, OR, NOT) or field tags
  - Express relationships naturally: "X in Y", "X role in Y", "X effects on Y"
"""

# Keyword-centric prompt section (only used when keywords are available)
QUERY_CONSTRUCTION_WITH_KEYWORDS = """
CONSTRUCTION STRATEGY - Keywords + Content:

Your query should combine:
1. EXTRACTED KEYWORDS - Statistical/algorithmic extraction provides structural guidance
2. FULL CONTENT (if available) - Additional context to refine and enhance the query

**How to Use Both:**
- KEYWORDS provide STRUCTURAL GUIDANCE: They identify the most statistically significant terms
- CONTENT provides SEMANTIC CONTEXT: It helps you understand relationships, expand synonyms, and add precision
- Your query should be ANCHORED by keywords but ENHANCED by content understanding
- Don't just concatenate keywords - use content to understand WHY they're important

**Example Workflow:**
1. Keywords: ["BRCA1", "mutation", "risk"]
2. Content reveals: "BRCA1 mutations confer hereditary breast cancer susceptibility"
3. Query: BRCA1[Title/Abstract] AND (hereditary OR familial) AND "breast cancer" AND (mutation OR variant)
   → Used keywords as anchors, content understanding added "hereditary", "familial", proper disease term

**Strategy:**
1. START WITH KEYWORDS as your FOUNDATION - they've been extracted for statistical significance
2. ENHANCE WITH CONTENT - use it to understand context, find synonyms, discover relationships
3. APPLY BACKEND-SPECIFIC SYNTAX as appropriate

**Keyword Scores:**
• Extractor name tells you HOW keywords were scored:
  - YAKE: Lower scores = more important (0.0 is perfect)
  - RAKE: Higher scores = more important
  - TF-IDF: Higher scores = more important
• Use this to prioritize which keywords to emphasize

GOOD EXAMPLES:

✅ Keywords: ["FBLN4", "calcification"], Content: "FBLN4 mutations cause arterial calcification"
   → Query: FBLN4[Title] AND ("vascular calcification" OR "arterial calcification")
   → Used keyword as anchor, content revealed "vascular" and "arterial" context

✅ Keywords: ["BMPR2", "hypertension", "familial"], Scores: [0.05, 0.12, 0.18], Extractor: yake
   → Query: BMPR2[Gene] AND ("pulmonary hypertension" OR "PAH") AND (familial OR hereditary)
   → Prioritized BMPR2 (best score), expanded hypertension with domain knowledge

AVOID:
❌ Ignoring keywords and just reading content → Keywords provide essential statistical signals
❌ Blindly concatenating keywords → Use content to understand relationships
❌ Using content instead of keywords → Content should ENHANCE keywords, not replace them
❌ Adding too many terms from content → Stay focused on core concepts from keywords
"""

# Content-only prompt section (used when no keywords are available)
QUERY_CONSTRUCTION_CONTENT_ONLY = """
CONSTRUCTION STRATEGY - Content-Based Query Generation:

Since no statistical keywords are provided, you must extract key concepts DIRECTLY from the resource content to build a precise query.

**Critical Requirements:**
1. DO NOT simply copy the title verbatim as your query
2. READ the full content (title + abstract/body) to identify:
   - Core entities (genes, proteins, diseases, organisms, cell types)
   - Key mechanisms or pathways
   - Specific phenotypes or outcomes
   - Technical terminology that makes this paper distinctive
3. EXPAND with domain knowledge:
   - Add synonyms and related terms
   - Include alternative names for genes/diseases
   - Consider different ways to express the same concept
4. FOCUS on SPECIFICITY:
   - Avoid generic terms ("cancer", "mutation", "protein" alone)
   - Combine multiple specific features
   - Target distinguishing characteristics

**Example Workflow:**
Content: "Loss-of-function mutations in BMPR2 cause familial pulmonary arterial hypertension through impaired TGF-β signaling in pulmonary vascular endothelial cells"
Title: "BMPR2 mutations in familial PAH"

BAD Query: "BMPR2 mutations in familial PAH"  ❌ (just copied the title)

GOOD Query: "BMPR2 pulmonary arterial hypertension familial hereditary TGF-beta signaling endothelial dysfunction"
✅ Extracted key concepts (BMPR2, PAH, familial), added synonyms (hereditary), included mechanism (TGF-beta), added cellular context (endothelial)

**Strategy:**
1. IDENTIFY core concepts from content (entities, mechanisms, contexts)
2. EXTRACT 5-10 most specific/distinctive terms
3. EXPAND with synonyms and related terminology
4. COMBINE into a focused query that balances precision and recall

GOOD EXAMPLES:

✅ Content about FBLN4 and vascular calcification
   → Query: "FBLN4 vascular calcification arterial stiffness elastin degradation"
   → Extracted gene, phenotype, mechanism - more specific than title alone

✅ Content about Marfan syndrome and fibrillin
   → Query: "Marfan syndrome fibrillin FBN1 aortic dissection connective tissue"
   → Combined disease, protein, gene symbol, key complication, tissue context

AVOID:
❌ Copying the title verbatim → No synonym expansion or enhancement
❌ Using only 1-2 terms → Too broad, won't distinguish this paper
❌ Adding too many generic terms → Dilutes specificity
❌ Ignoring the abstract/body content → Missing key context and mechanisms
"""


# Combined prompt factory function (to be used in code)
def build_query_construction_prompt(has_keywords: bool) -> str:
    """
    Build dynamic system prompt based on whether keywords are available.

    Parameters:
        has_keywords: bool - Whether keywords were extracted

    Returns:
        str - Complete system prompt
    """
    if has_keywords:
        return (
            QUERY_CONSTRUCTION_BASE
            + QUERY_CONSTRUCTION_WITH_KEYWORDS
            + """
YOUR OUTPUT:
Generate ONE query that balances:
- Keyword-driven structure (use the extracted keywords as anchors)
- Content-informed enhancement (when content available, understand context and add synonyms/precision)
- Backend-appropriate syntax (if configured)
Explain your construction strategy: how did you use keywords AND content together?
"""
        )
    else:
        return (
            QUERY_CONSTRUCTION_BASE
            + QUERY_CONSTRUCTION_CONTENT_ONLY
            + """
YOUR OUTPUT:
Generate ONE query that balances:
- Content-driven concept extraction (identify key entities, mechanisms, contexts)
- Domain knowledge enhancement (synonyms, related terms, alternative names)
- Backend-appropriate syntax (if configured)
Explain your construction strategy: what key concepts did you extract and why?
"""
        )


# Deprecated - kept for backwards compatibility but will be removed
QUERY_CONSTRUCTION_PROMPT = build_query_construction_prompt(has_keywords=True)

# ==============================================================================
# Query Generation Prompts (End-to-end, combines Stage 1+2)
# ==============================================================================

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

4. QUERY DIVERSITY
   • If generating multiple queries, use different angles/perspectives
   • Query 1: Focus on main entities and their relationship
   • Query 2: Focus on biological mechanism or pathway
   • Query 3: Focus on context, disease, or therapeutic angle

5. AVOID THESE MISTAKES
   ❌ Single broad terms: "cancer", "mutation", "gene"
   ❌ Too many OR clauses: matches too many papers
   ❌ Generic phrases: "plays a role", "is associated with"
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

4. QUERY DIVERSITY
   • If generating multiple queries, use different angles/perspectives
   • Query 1: Focus on main entities and their relationship
   • Query 2: Focus on biological mechanism or pathway
   • Query 3: Focus on context, disease, or therapeutic angle

5. AVOID THESE MISTAKES
   ❌ Single broad terms: "cancer", "mutation", "gene"
   ❌ Too many OR clauses: matches too many papers
   ❌ Generic phrases: "plays a role", "is associated with"
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
