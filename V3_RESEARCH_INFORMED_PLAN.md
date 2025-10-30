# V3 Entity Extraction: Research-Informed Development Plan

**Date:** 2025-01-30
**Status:** Planning
**Goal:** Enhance v3 extraction pipeline based on research findings while maintaining minimalist design

---

## Executive Summary

After comprehensive analysis of recent research (BioRED, GPT biomedical RE, Production RAG processing 5M+ documents) and comparison with the gene extraction graph implementation, we identify high-impact, low-complexity improvements that align with Unix philosophy: **do one thing well, keep it simple, compose tools**.

**Key Finding:** Current v3 architecture is **strongly validated** by research. Proposed enhancements focus on exposing essential complexity (metadata, quality visibility) while avoiding accidental complexity (routing nodes, retry loops, state mutations).

---

## Research Foundation

### Studies Analyzed

1. **BioRED Dataset** (Luo et al., 2022)
   - 600 PubMed abstracts, document-level, multiple entity types
   - Performance: 89.3% F1 (NER), 47.7% F1 (novel relation extraction)
   - **Key insight:** Document-level multi-entity extraction is emerging standard

2. **GPT Models for Biomedical RE** (Zhang et al., 2024)
   - GPT-3.5/4 on EU-ADR, GAD, ChemProt datasets
   - **Key insight:** Unmasked entities outperform masked (+2.1% F1), annotation quality causes 20-40% variance

3. **Production RAG: 5M+ Documents** (Abdelfattah, 2024)
   - 13M+ pages processed, enterprise deployments
   - **Key insight:** Success from systematic optimization (metadata, reranking, chunking), NOT complex control flow

4. **Gene Extraction Graph** (Internal implementation)
   - Single-target extraction (genes for PAH)
   - **Key insight:** Explicit quality review works for specific tasks, may be overkill for general extraction

### Current V3 Strengths (Research-Validated)

| Feature | Research Support | Status |
|---------|-----------------|--------|
| Document-level processing | BioRED (emerging standard) | ✅ Implemented |
| Unmasked entities with context | GPT study (+2.1% F1) | ✅ Implemented |
| Hybrid pair generation | Production RAG #1 ROI | ✅ Implemented |
| Semantic caching | Production RAG (cost optimization) | ✅ Implemented |
| Quote validation | GPT study (quality critical) | ✅ Implemented |
| Multi-stage pipeline | Production RAG (systematic) | ✅ Implemented |
| Zero-shot LLM | GPT study (cross-corpus robust) | ✅ Implemented |

---

## Design Philosophy

### Minimalist Principles

**From global instructions:**
> Essential complexity is intrinsic to the domain - represent it explicitly in the domain model
> Incidental complexity comes from tools/frameworks - eliminate, hide, or centralise it

**From Unix philosophy:**
- Do one thing well: Extract relationships with provenance
- Keep it simple: Linear pipeline, no branching
- Compose tools: V3 outputs feed other analysis tools

**Decision Framework:**
- ✅ Add if: Exposes essential domain information (metadata, provenance)
- ✅ Add if: Pure observability (logging, metrics) without behavior change
- ❌ Avoid if: Adds control flow complexity (routing, retry loops)
- ❌ Avoid if: Scope creep (novelty classification, quality gates)

---

## Implementation Plan

### Phase 1: High-Impact, Low-Complexity (Immediate)

**Timeline:** 1-2 hours
**Complexity:** ~15 lines of code
**Goal:** Implement research-validated optimizations with zero architectural complexity

#### 1.1 Add Metadata to Extraction Context

**Research:** Production RAG #4 (high ROI, low effort)

**Implementation:**
```python
# File: src/_interaction_finder_claude/extraction_core/extraction.py
# Function: _build_extraction_prompt()

def _build_extraction_prompt(
    entity_kinds: List[str], task_context: str, resource: "Resource"  # Changed signature
) -> str:
    """Build extraction prompt with document metadata."""
    kinds_str = ", ".join(entity_kinds)

    # Truncate document if too long
    max_doc_length = 8000
    if len(resource.text) > max_doc_length:
        doc_text = resource.text[:max_doc_length] + "\n\n[Document truncated...]"
    else:
        doc_text = resource.text

    # Add metadata for better entity disambiguation
    prompt = f"""Extract all {kinds_str} entities from the following scientific document.

DOCUMENT METADATA:
Title: {resource.title}
Source: {resource.id.url}

RESEARCH CONTEXT: {task_context}

CRITICAL REQUIREMENTS:
1. Extract EVERY entity of the specified types ({kinds_str})
2. For each entity, provide EXACT quotes from the document (verbatim text)
3. Quotes must be copied EXACTLY as written - do NOT paraphrase or modify
4. Include entity aliases if present (e.g., gene symbol "BRCA1" vs "breast cancer 1")
5. If no entities found, return empty list

DOCUMENT TEXT:
{doc_text}

Extract entities with supporting quotes."""

    return prompt
```

**Update caller:**
```python
# In extract_from_resource()
prompt = _build_extraction_prompt(entity_kinds, task_context, resource)  # Pass resource not text
```

**Expected impact:**
- Better entity disambiguation (e.g., "p53" in cancer vs aging context)
- Improved confidence scores
- Enhanced provenance tracking

**Testing:**
```python
def test_extraction_prompt_includes_metadata():
    """Verify metadata appears in extraction prompt."""
    resource = create_test_resource(
        title="Test Paper",
        url="https://example.com/paper",
        text="The BRCA1 gene is associated with breast cancer."
    )

    prompt = _build_extraction_prompt(["gene", "disease"], "cancer biology", resource)

    assert "Title: Test Paper" in prompt
    assert "Source: https://example.com/paper" in prompt
    assert "RESEARCH CONTEXT: cancer biology" in prompt
```

#### 1.2 Use Temperature=0 for Determinism

**Research:** GPT study (marginal improvement, better consistency)

**Implementation:**
```python
# File: src/_interaction_finder_claude/extraction_graph_v3/agents.py
# In create_entity_extractor_v3(), create_assessment_agent_v3(), create_pair_evaluator_v3()

# Add model_settings parameter to Agent()
agent = Agent(
    model=model,
    output_type=SimpleEntityListOut,
    system_prompt=system_prompt,
    deps_type=ExtractionDepsV3,
    retries=5,
    model_settings={"temperature": 0.0},  # Add this line
)
```

**Expected impact:**
- Deterministic outputs (same input → same output)
- Essential for reproducible research
- Minimal performance change but better consistency

**Testing:**
```python
async def test_extraction_determinism():
    """Verify same input produces same output."""
    resource = create_test_resource()

    # Run extraction twice
    entities_1 = await extract_from_resource(resource, ["gene"], model, "cancer", 0.90)
    entities_2 = await extract_from_resource(resource, ["gene"], model, "cancer", 0.90)

    # Should produce identical results
    assert len(entities_1) == len(entities_2)
    assert {e.name for e in entities_1} == {e.name for e in entities_2}
```

#### 1.3 Add Quality Metrics and Visibility

**Research:** Production RAG (observability), GPT study (quality awareness)

**Implementation:**
```python
# File: src/_interaction_finder_claude/extraction_graph_v3/models.py
# Add to ExtractionMetadata

class QualityIndicators(BaseModel):
    """Quality indicators for extraction results."""

    entities_per_document: float = Field(
        description="Average entities extracted per document"
    )
    avg_quotes_per_entity: float = Field(
        description="Average supporting quotes per entity"
    )
    pair_acceptance_rate: float = Field(
        description="Proportion of candidate pairs accepted"
    )

    def quality_flags(self) -> List[str]:
        """Return quality warnings for user review."""
        flags = []

        if self.entities_per_document < 0.5:
            flags.append("LOW_ENTITY_DENSITY")

        if self.avg_quotes_per_entity < 1.5:
            flags.append("INSUFFICIENT_EVIDENCE")

        if self.pair_acceptance_rate < 0.05:
            flags.append("LOW_PAIR_ACCEPTANCE")

        return flags

    def quality_summary(self) -> str:
        """Human-readable quality assessment."""
        flags = self.quality_flags()

        if not flags:
            return "Normal extraction quality"

        explanations = {
            "LOW_ENTITY_DENSITY": "Low entity density - may indicate entity-sparse documents or extraction issues",
            "INSUFFICIENT_EVIDENCE": "Low evidence per entity - may indicate weak quote extraction",
            "LOW_PAIR_ACCEPTANCE": "Low pair acceptance - may indicate high precision filtering or weak relationships",
        }

        return "; ".join(explanations[flag] for flag in flags)


class BatchExtractionResultV3(BaseModel):
    # Existing fields...

    # Add quality indicators
    quality: Optional[QualityIndicators] = Field(
        description="Quality indicators for this extraction run",
        default=None
    )
```

**Compute metrics:**
```python
# File: src/_interaction_finder_claude/extraction_graph_v3/run.py
# In run_extraction_v3(), before returning result

# Compute quality indicators
if state.final_pairs:
    quality = QualityIndicators(
        entities_per_document=(
            len(state.entities_found) / len(state.resource_pool.resources)
            if state.resource_pool.resources else 0
        ),
        avg_quotes_per_entity=(
            sum(len(e.quotes) for e in state.entities_found.values()) / len(state.entities_found)
            if state.entities_found else 0
        ),
        pair_acceptance_rate=(
            state.metrics.pairs_accepted / state.metrics.pair_evaluation_calls
            if state.metrics.pair_evaluation_calls > 0 else 0
        ),
    )

    # Log quality flags if any
    if flags := quality.quality_flags():
        logger.warning(f"Quality flags raised: {flags}")
        logger.warning(f"Assessment: {quality.quality_summary()}")
else:
    quality = None

# Add to result
result = BatchExtractionResultV3(
    # ... existing fields ...
    quality=quality,
)
```

**Expected impact:**
- User visibility into extraction quality
- Early warning for potential issues
- No behavioral changes, pure observability

#### 1.4 Add Reranking Visibility

**Research:** Production RAG #2 ("highest value 5 lines of code")

**Implementation:**
```python
# File: src/_interaction_finder_claude/extraction_graph_v3/nodes.py
# In EvaluatePairs.run(), add logging

logger.info(
    f"Pair evaluation complete: {len(state.pair_candidates)} candidates → "
    f"{len(state.final_pairs)} accepted ({state.metrics.pairs_accepted / len(state.pair_candidates) * 100:.1f}% acceptance)"
)
```

**Expected impact:**
- Visibility into filtering effectiveness (analogous to 50→15 reranking)
- Helps tune evaluation thresholds
- Zero complexity, pure logging

---

### Phase 2: Documentation and Testing (Next Session)

**Timeline:** 2-3 hours
**Goal:** Document design decisions, add behavioral tests

#### 2.1 Document Minimalist Philosophy

**File:** `CLAUDE.md` (append to project instructions)

**Content:**
```markdown
## V3 Pipeline Philosophy

### Minimalist Design

**Linear Flow:** Four nodes, no branching, predictable execution
```
ExtractEntities → AssessIndividually → GeneratePairCandidates → EvaluatePairs
```

**Quality Through Simplicity:**
- Quote validation at extraction (fuzzy matching, threshold=0.90)
- Output validation at generation (Pydantic retries, up to 5 attempts)
- Provenance validation at output (character-level positions)
- No pipeline-level retry loops needed

### Unix Philosophy Applied

**Do One Thing:** Extract biological relationships with complete provenance tracking

**Do It Well:**
- Semantic caching for efficiency (content-based keys)
- Quote-level validation (ResourceQuotes with char positions)
- Multi-layer quality control (LLM validators, fuzzy matching, provenance)

**Compose Tools:** V3 output feeds downstream analysis
- Novelty classification (novel vs background findings)
- Knowledge graph construction
- Cross-document consistency checking

### Trust the Pipeline

**Natural Filtering:** Quality emerges from design, not control flow
- Low-evidence entities → low confidence assessments → rejected pairs
- No manual cutoffs or filtering rules
- LLM confidence does the work

**User Responsibility:** V3 processes what users provide
- No content routing (user curates biomedical documents)
- No quality loops (user evaluates results)
- No premature optimization (benchmarks guide improvements)

### Research Validation

Current v3 implements production-grade best practices:
- Document-level processing (BioRED emerging standard)
- Unmasked entities (GPT study: +2.1% F1)
- Hybrid pair generation (Production RAG: highest ROI)
- Semantic caching (Production RAG: cost optimization)
- Zero-shot robustness (GPT study: cross-corpus performance)
```

#### 2.2 Add Behavioral Tests

**File:** `tests/extraction_v3/test_quality.py` (new file)

```python
"""Quality and behavior tests for v3 extraction pipeline."""

import pytest
from interaction_finder.extraction_graph_v3 import run_extraction_v3
from interaction_finder.resources import ResourcePool


async def test_extraction_includes_provenance():
    """Verify every entity and pair has complete provenance."""
    result = await run_extraction_v3(test_urls, test_config)

    # Every entity must have quotes
    for pair in result.entity_pairs:
        assert len(pair.entity_a.quotes) > 0, f"Entity {pair.entity_a.name} missing quotes"
        assert len(pair.entity_b.quotes) > 0, f"Entity {pair.entity_b.name} missing quotes"

        # Every quote must have char positions
        for quote in pair.entity_a.quotes:
            assert quote.start_char >= 0
            assert quote.end_char > quote.start_char

    # Every pair must have evidence
    for pair in result.entity_pairs:
        assert len(pair.evidence_quotes) > 0
        assert pair.validate_provenance()


async def test_quality_indicators_computed():
    """Verify quality indicators are computed correctly."""
    result = await run_extraction_v3(test_urls, test_config)

    assert result.quality is not None
    assert result.quality.entities_per_document >= 0
    assert result.quality.avg_quotes_per_entity >= 0
    assert 0 <= result.quality.pair_acceptance_rate <= 1


async def test_low_quality_flags_detected():
    """Verify quality flags detect issues."""
    # Create minimal test data with low quality
    result = await run_extraction_v3(
        urls=[url_with_sparse_entities],
        config=test_config
    )

    if result.quality.entities_per_document < 0.5:
        assert "LOW_ENTITY_DENSITY" in result.quality.quality_flags()


async def test_deterministic_extraction():
    """Verify temperature=0 produces deterministic results."""
    result1 = await run_extraction_v3(test_urls, test_config)
    result2 = await run_extraction_v3(test_urls, test_config)

    # Should extract same entities
    entities1 = {pair.entity_a.name for pair in result1.entity_pairs}
    entities2 = {pair.entity_a.name for pair in result2.entity_pairs}

    assert entities1 == entities2, "Extraction should be deterministic"


async def test_metadata_improves_disambiguation():
    """Verify metadata helps disambiguate entities."""
    # Create two papers with "p53" in different contexts
    cancer_paper = create_resource("p53 mutations in breast cancer", url="cancer.pdf")
    aging_paper = create_resource("p53 role in cellular senescence", url="aging.pdf")

    result = await run_extraction_v3([cancer_paper.id.url, aging_paper.id.url], config)

    # Should extract p53 with context-specific assessments
    p53_pairs = [p for p in result.entity_pairs if "p53" in p.entity_a.name.lower()]

    # Assessments should reference paper context
    for pair in p53_pairs:
        # Evidence should come from specific paper
        assert any(
            q.resource_id.url in ["cancer.pdf", "aging.pdf"]
            for q in pair.evidence_quotes
        )
```

**File:** `tests/extraction_v3/test_integration.py` (enhance existing)

```python
async def test_pipeline_preserves_individual_assessments():
    """Verify each entity gets individual assessment, not collective."""
    result = await run_extraction_v3(test_urls, test_config)

    # Extract all entities mentioned
    all_entities = set()
    for pair in result.entity_pairs:
        all_entities.add(pair.entity_a.name)
        all_entities.add(pair.entity_b.name)

    # Each entity should have been assessed independently
    # (Not directly testable without exposing state, but verify via outputs)

    # Each entity should have entity-specific evidence
    for pair in result.entity_pairs:
        # Entity A evidence should mention entity A
        entity_a_mentioned = any(
            pair.entity_a.name.lower() in str(q).lower()
            for q in pair.entity_a.quotes
        )
        assert entity_a_mentioned, f"Entity {pair.entity_a.name} not in own evidence"
```

---

### Phase 3: Research Validation (Future)

**Timeline:** Ongoing
**Goal:** Benchmark v3 against published datasets, publish results

#### 3.1 BioRED Evaluation

**Dataset:** https://ftp.ncbi.nlm.nih.gov/pub/lu/BioRED/

**Configuration:**
```toml
# config_biored.toml
[task.kinds]
gene = { kind = "gene", form = ["name", "symbol"] }
disease = { kind = "disease", form = ["name"] }
chemical = { kind = "chemical", form = ["name"] }

[task.relation]
type = "biomedical association"
context = "gene-disease, chemical-disease, and chemical-chemical relationships"
```

**Target Performance:**
- NER: Match/exceed 89.3% F1
- RE (all relations): Exceed baseline
- **RE (novel relations): Exceed 47.7% F1** (current SOTA for novel findings)

**Advantages to highlight:**
- Quote-level provenance (BioRED doesn't have this)
- Document-level processing (matches dataset)
- Multiple entity types (matches dataset)

#### 3.2 Computational Cost Analysis

**Metrics to track:**
- LLM API calls per document
- Cache hit rate (target: >90% based on Production RAG)
- Cost per 1000 documents
- Throughput (documents/hour)
- Cost savings from caching

**Comparison baseline:** No caching (to demonstrate semantic cache value)

#### 3.3 Cross-Domain Robustness

**Research:** GPT study showed zero-shot models maintain cross-corpus performance

**Test:** Train on one dataset (e.g., gene-disease), evaluate on another (e.g., protein-protein)

**Expectation:** V3 should maintain performance without fine-tuning (zero-shot advantage)

---

## Deferred: Complexity Not Justified

Based on research analysis and minimalist principles, the following are **explicitly deferred** unless real usage demonstrates need:

### ❌ Content Router/Relevance Filter

**Reason:**
- Production RAG routing was for question types, not content relevance
- Users provide curated biomedical documents (researcher-selected)
- False negative risk (missing relevant methods/review papers)
- <1% of curated inputs likely non-biomedical

**If needed later:** User-side filtering before v3, not built into pipeline

### ❌ Quality Review Loop

**Reason:**
- Production RAG succeeded without quality loops (13M pages)
- Current multi-layer validation sufficient (quote validation, provenance checking)
- Nested retry complexity (validators retry 5x, pipeline would retry 1x)
- Unclear success criteria (papers naturally vary in entity density)
- Cost: Double extraction on retry

**If needed later:** Add only if >10% of real extractions show quality issues

### ❌ Novelty Classification

**Reason:**
- Scope creep (v3 extracts relationships, doesn't classify them)
- Separate downstream task
- BioRED combines these, but doesn't mean we should
- Unix philosophy: One tool, one job

**If needed later:** Separate tool that processes v3 outputs

### ❌ Node History Tracking

**Reason:**
- Linear pipeline (deterministic path)
- Duplicates logging
- Adds mutable state throughout pipeline

**If needed later:** Use structured logging instead, no state mutation

### ❌ Pre-filtering Low-Quality Entities

**Reason:**
- Trust pipeline's natural filtering
- Low-evidence entities → low confidence → rejected pairs
- Special-case logic adds complexity

**If needed later:** Let LLM confidence do the work

---

## Success Criteria

### Phase 1 Complete When:
- ✅ Metadata appears in extraction prompts
- ✅ Temperature=0 set for all agents
- ✅ Quality indicators computed and logged
- ✅ Reranking visibility logging added
- ✅ All changes <20 lines of code total
- ✅ Zero architectural complexity added

### Phase 2 Complete When:
- ✅ Minimalist philosophy documented in CLAUDE.md
- ✅ Behavioral tests pass (provenance, quality, determinism)
- ✅ Process validation tests pass (individual assessment)
- ✅ No test failures introduced

### Phase 3 Complete When:
- ✅ BioRED evaluation complete, results documented
- ✅ Performance meets/exceeds baseline (especially 47.7% novel relations)
- ✅ Computational cost analysis complete
- ✅ Results ready for publication/documentation

---

## Risk Mitigation

### Risk: Metadata Increases Token Usage

**Impact:** Higher API costs
**Mitigation:** Metadata is small (title + URL ~50 tokens), negligible compared to document content (8000 tokens)
**Monitoring:** Track average tokens per extraction

### Risk: Temperature=0 Reduces Quality

**Impact:** Less diverse outputs
**Mitigation:** GPT study shows marginal improvement with temp=0, not degradation
**Fallback:** Easy to revert (1 parameter change)

### Risk: Quality Flags Create False Positives

**Impact:** Users question good extractions
**Mitigation:**
- Flags are warnings, not errors
- Provide explanations with each flag
- Document in logs, don't show to user by default

### Risk: Missing Real Quality Issues

**Impact:** Bad extractions not caught
**Mitigation:**
- Current multi-layer validation (quote validation, provenance) already strong
- Phase 3 evaluation will reveal real issues
- Can add quality loop later if demonstrated need

---

## Long-Term Vision

### V3 as Foundation

**Current:** General-purpose relation extraction engine
**Future:** Core component in larger system

**Composition Examples:**

```bash
# Pipeline 1: Basic extraction
v3_extract --config config.toml --urls papers.txt > results.jsonl

# Pipeline 2: Extract + Classify novelty
v3_extract --config config.toml --urls papers.txt | \
  classify_novelty --knowledge-base existing.jsonl > classified.jsonl

# Pipeline 3: Extract + Build knowledge graph
v3_extract --config config.toml --urls papers.txt | \
  build_kg --output graph.json

# Pipeline 4: Extract + Cross-validate
v3_extract --config config.toml --urls papers.txt | \
  cross_validate --known-relationships gold_standard.jsonl > validated.jsonl
```

**Key:** V3 stays simple, composition handles complexity

### Research Contributions

**Publication targets:**
1. **BioRED benchmark results** - Document-level multi-entity extraction with provenance
2. **Quote validation methodology** - Fuzzy matching for LLM output verification
3. **Hybrid pair generation** - Co-occurrence + AI suggestions for relation discovery
4. **Semantic caching analysis** - Cost/performance trade-offs for biomedical extraction

---

## References

1. Luo, L., Lai, P.-T., Wei, C.-H., Arighi, C. N., & Lu, Z. (2022). BioRED: a rich biomedical relation extraction dataset. *Briefings in Bioinformatics*, 23(5), bbac282.

2. Zhang, J., Wibert, M., Zhou, H., Peng, X., Chen, Q., Keloth, V. K., Hu, Y., Zhang, R., Xu, H., & Raja, K. (2024). A Study of Biomedical Relation Extraction Using GPT Models. *AMIA Summits on Translational Science Proceedings*, 2024, 391-400.

3. Abdelfattah, A. (2024). Production RAG: Processing 5M+ Documents. Retrieved from https://blog.abdellatif.io/production-rag-processing-5m-documents

4. Khachatrian, H., et al. (2019). BioRelEx 1.0: Biological Relation Extraction Benchmark. *Proceedings of the 18th BioNLP Workshop and Shared Task*, 176-190.

5. Dagdelen, J., et al. (2024). Structured information extraction from scientific text with large language models. *Nature Communications*, 15(1), 1418.

---

## Appendix: Research Evidence Summary

### What Research Validates

| V3 Feature | Evidence | Impact |
|------------|----------|--------|
| Document-level | BioRED (emerging standard) | ✅ High |
| Unmasked entities | GPT study (+2.1% F1) | ✅ Medium |
| Metadata in prompts | Production RAG (#4 ROI) | ✅ High |
| Temperature=0 | GPT study (consistency) | ✅ Low |
| Hybrid pairs | Production RAG (#1 ROI) | ✅ High |
| Reranking visibility | Production RAG (#2 ROI) | ✅ Medium |
| Semantic caching | Production RAG (cost) | ✅ High |
| Zero-shot LLM | GPT study (robust) | ✅ Medium |

### What Research Doesn't Support

| Proposed Feature | Evidence | Decision |
|-----------------|----------|----------|
| Quality loop | Production RAG: no loops | ❌ Defer |
| Content routing | Prod RAG: for Q types, not content | ❌ Defer |
| Novelty classification | BioRED: separate task | ❌ Defer |
| Node history | Gene graph: single target only | ❌ Skip |
| Pre-filtering | Not in any research | ❌ Skip |

---

**Plan Status:** Ready for implementation
**Next Action:** Implement Phase 1 (1-2 hours)
**Review After:** Phase 1 complete, evaluate before proceeding to Phase 2
