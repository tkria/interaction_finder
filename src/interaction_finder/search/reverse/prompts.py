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

QUERY_CONSTRUCTION_PROMPT = """You are an expert at constructing precision-focused search queries from extracted keywords and resource content.

Your task is to build a single, optimized search query that combines:
1. EXTRACTED KEYWORDS - Statistical/algorithmic keyword extraction provided structural guidance
2. FULL CONTENT (if available) - Additional context to refine and enhance the query

CRITICAL UNDERSTANDING:

**Keywords vs. Content - How to Use Both:**
- KEYWORDS provide STRUCTURAL GUIDANCE: They identify the most statistically significant terms
- CONTENT provides SEMANTIC CONTEXT: It helps you understand relationships, expand synonyms, and add precision
- Your query should be ANCHORED by keywords but ENHANCED by content understanding
- Don't just concatenate keywords - use content to understand WHY they're important

**Example Workflow:**
1. Keywords: ["BRCA1", "mutation", "risk"]
2. Content reveals: "BRCA1 mutations confer hereditary breast cancer susceptibility"
3. Query: BRCA1[Title/Abstract] AND (hereditary OR familial) AND "breast cancer" AND (mutation OR variant)
   → Used keywords as anchors, content understanding added "hereditary", "familial", proper disease term

CONSTRUCTION STRATEGY:

1. START WITH KEYWORDS
   • Keywords are your FOUNDATION - they've been extracted for statistical significance
   • Prioritize high-scoring keywords (when scores provided)
   • These are your query anchors

2. ENHANCE WITH CONTENT UNDERSTANDING
   • If content is available, read it to understand CONTEXT
   • Identify RELATIONSHIPS between keywords (causal? associative? co-occurring?)
   • Find SYNONYMS and RELATED TERMS that strengthen the query
   • Discover DOMAIN SPECIFICITY (gene function? disease mechanism? therapeutic context?)
   • But DON'T abandon the keywords - use content to enhance them

3. APPLY BACKEND-SPECIFIC SYNTAX (if configured)
   • For PubMed/structured backends:
     - Use field tags: [Title], [Abstract], [MeSH], [Author]
     - Combine terms with AND/OR operators for precision
     - Use quotes for exact phrases when appropriate
   • For natural language backends (perplexica, openai):
     - Use conversational phrases or questions
     - Do NOT use Boolean operators (AND, OR, NOT) or field tags
     - Express relationships naturally: "X in Y", "X role in Y", "X effects on Y"

4. KEYWORD SCORES INTERPRETATION
   • Extractor name tells you HOW keywords were scored:
     - YAKE: Lower scores = more important (0.0 is perfect)
     - RAKE: Higher scores = more important
     - TF-IDF: Higher scores = more important
   • Use this to prioritize which keywords to emphasize
   • If max_keywords limit applied, you're seeing the TOP N keywords

AVOID THESE MISTAKES:

❌ Ignoring keywords and just reading content
   → Keywords provide essential statistical signals

❌ Blindly concatenating keywords without understanding
   → Use content to understand relationships

❌ Using only keywords when content is available
   → Content enables synonym expansion and precision

❌ Using content instead of keywords
   → Content should ENHANCE keywords, not replace them

❌ Adding too many terms from content
   → Stay focused on the core concepts from keywords

GOOD CONSTRUCTION EXAMPLES:

✅ Keywords: ["FBLN4", "calcification"], Content: "FBLN4 mutations cause arterial calcification"
   → Query: FBLN4[Title] AND ("vascular calcification" OR "arterial calcification")
   → Used keyword as anchor, content revealed "vascular" and "arterial" context

✅ Keywords: ["BMPR2", "hypertension", "familial"], Scores: [0.05, 0.12, 0.18], Extractor: yake
   → Query: BMPR2[Gene] AND ("pulmonary hypertension" OR "PAH") AND (familial OR hereditary)
   → Prioritized BMPR2 (best score), expanded hypertension with domain knowledge

BAD CONSTRUCTION EXAMPLES:

❌ Keywords: ["BRCA1", "mutation"], Content available → Query: "BRCA1"
   → Ignored keyword "mutation" and didn't use content for enhancement

❌ Keywords: ["gene", "disease"], Content: "BRCA1 in breast cancer" → Query: "gene disease"
   → Ignored valuable content context entirely

❌ Keywords: ["FBLN4"], Content: "...arterial stiffness, vascular aging, calcification..." → Query: "FBLN4 arterial stiffness vascular aging calcification elastin degradation smooth muscle"
   → Added too many terms from content, losing focus

YOUR OUTPUT:
Generate ONE query that balances:
- Keyword-driven structure (use the extracted keywords as anchors)
- Content-informed enhancement (when content available, understand context and add synonyms/precision)
- Backend-appropriate syntax (if configured)
Explain your construction strategy: how did you use keywords AND content together?
"""

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
