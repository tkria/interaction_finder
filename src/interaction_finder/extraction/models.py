"""Data models for the extraction pipeline.

This module defines:
1. Core data structures (dataclasses for State-internal, Pydantic for serialization)
2. Agent output models (Pydantic) returned by LLM agents
3. Final pipeline output models (Pydantic)

The separation ensures clean boundaries: agents return Pydantic models with
raw string quotes, which are immediately converted to ResourceQuote objects
for storage in State. Models that appear in serialized output use Pydantic BaseModel
to enable proper JSON serialization via model_dump().
"""

from dataclasses import dataclass
from typing import ClassVar, Iterable, Literal, NamedTuple

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator
from pydantic.functional_validators import SkipValidation
from pydantic_core import to_jsonable_python

from interaction_finder.resources import ResourceId, ResourcePool, ResourceQuote

# =============================================================================
# Core data structures (used in State)
# =============================================================================


class EntityMention(BaseModel):
    """Entity found in a single resource (after quote validation).

    This is the validated, resource-quote-enriched version of EntityInfo.
    Multiple EntityInfo instances with the same name are merged during conversion.

    Attributes:
        kind: Entity type (e.g., "gene", "disease", "protein")
        name: Canonical entity name (e.g., "BRCA1")
        aliases: Names as they appear in text (e.g., ["BRCA-1", "BRCA1"])
        quotes: Supporting quotes from the resource (validated ResourceQuote objects)
        reasoning: Explanation of extraction choices (merged if duplicates combined)
    """

    kind: str
    name: str
    aliases: list[str]
    quotes: SkipValidation[
        list[ResourceQuote]
    ]  # Skip validation to avoid Resource copying
    reasoning: str


@dataclass
class ProximalEntitySet:
    """Group of entities found in close proximity within a document.

    Used to identify regions where multiple entities co-occur, which may
    indicate potential associations worth investigating.

    Attributes:
        entities: Set of canonical entity names found in proximity
        chunk_range: (start_chunk_idx, end_chunk_idx) spanning all quotes
        entity_quotes: Mapping from canonical name to quotes for that entity
    """

    entities: set[str]
    chunk_range: tuple[int, int]
    entity_quotes: dict[str, list[ResourceQuote]]


class EntityPairKey(NamedTuple):
    """Identifier for an entity pair across documents.

    Entity names are ordered lexicographically by their kinds to ensure
    consistent pairing (e.g., always gene-disease, not disease-gene).

    Attributes:
        entity1_name: First entity canonical name
        entity2_name: Second entity canonical name
    """

    entity1_name: str
    entity2_name: str


class EntityRef(BaseModel):
    """Reference to an entity with mutable canonical name and provenance.

    Separates identity (canonical name, updated by consolidation) from provenance
    (original extractions, preserved as mentions). This enables tracing what was
    originally extracted vs what resulted from consolidation decisions.

    During JSON serialization, EntityRef is replaced with just the canonical name
    string. The full entity data is collected in an 'entities' dict at the
    checkpoint level, similar to how ResourceQuote references the ResourcePool.

    Attributes:
        canonical: Current canonical name (may be updated by consolidation rules)
        mentions: Original EntityMention(s) from extraction (append-only)
    """

    canonical: str
    mentions: list[EntityMention] = []

    @model_validator(mode="before")
    @classmethod
    def _accept_canonical_string(cls, data):
        """Accept canonical string and create minimal EntityRef."""
        if isinstance(data, str):
            return {"canonical": data, "mentions": []}
        return data

    @model_serializer(mode="wrap")
    def _serialize(self, serializer, info):
        """Replace full EntityRef with canonical string during JSON serialization."""
        if info.mode == "json":
            return self.canonical
        return serializer(self)

    def aliases(self) -> list[str]:
        """Aggregate aliases from all mentions (ordered, deduped)."""
        seen: set[str] = set()
        result: list[str] = []
        for mention in self.mentions:
            # Include original name if it differs from canonical
            if mention.name != self.canonical and mention.name not in seen:
                result.append(mention.name)
                seen.add(mention.name)
            for alias in mention.aliases:
                if alias not in seen:
                    result.append(alias)
                    seen.add(alias)
        return result

    def quotes(self) -> list[ResourceQuote]:
        """Aggregate quotes from all mentions."""
        aggregated: list[ResourceQuote] = []
        for mention in self.mentions:
            aggregated.extend(mention.quotes)
        return aggregated

    def reasoning(self) -> str:
        """Merge reasoning from all mentions."""
        if not self.mentions:
            return ""
        parts = []
        for mention in self.mentions:
            if mention.name == self.canonical:
                parts.append(mention.reasoning)
            else:
                parts.append(f"MERGED({mention.name}): {mention.reasoning}")
        return " | ".join(parts)

    @property
    def kind(self) -> str:
        """Kind is taken from the first mention."""
        if not self.mentions:
            raise ValueError("EntityRef has no mentions to derive kind")
        return self.mentions[0].kind


# =============================================================================
# Agent output models (Pydantic - what LLMs return)
# =============================================================================


class EntityInfo(BaseModel):
    """LLM output for a single entity.

    This is converted to EntityMention after quote validation.
    """

    model_config = ConfigDict(populate_by_name=True)

    kind: str = Field(
        alias="type", description='Entity type (e.g., "gene", "disease", "protein")'
    )
    name: str = Field(description="Canonical name of the entity")
    aliases: list[str] = Field(
        description=(
            "Lexical variants, acronyms, and synonyms as they appear verbatim in text. "
            "Include only names that refer to THIS entity, not related subtypes or complications."
        ),
        min_length=1,
    )
    quotes: list[str] = Field(
        description="Direct quotes from text supporting this entity", min_length=1
    )
    reasoning: str = Field(
        min_length=20, description="Brief explanation of why this entity was extracted"
    )


class EntityExtractionOut(BaseModel):
    """LLM output: all entities found in document (legacy, use DocumentAnalysisOut)."""

    entities: list[EntityInfo] = Field(description="Entities found in document")


class DocumentAnalysisOut(BaseModel):
    """Combined paper quality assessment and entity extraction.

    Field order is intentional: quality assessment comes first to ensure
    the model evaluates the paper holistically before extracting entities.
    This provides context for extraction and catches low-quality papers early.

    The quality assessment acts as a "reading comprehension" pass, forcing
    the model to engage with methodology, data, and claims before identifying
    entities relevant to the research topic.
    """

    # === Quality assessment (first: establishes paper understanding) ===
    paper_quality: "PaperQualityAssessment" = Field(
        description="Assessment of paper quality using seven-dimension rubric (complete this BEFORE extracting entities)"
    )
    # === Entity extraction (last: informed by quality assessment) ===
    entities: list[EntityInfo] = Field(
        description="Entities found in document (may be empty if paper has no relevant entities)"
    )


class EntityConsolidationDecision(BaseModel):
    """LLM decision to merge or rename an entity pair.

    Used when one entity name is a substring/similar to another (e.g., "BRCA" vs "BRCA1").
    Pairs not returned are implicitly skipped (kept separate).

    The pair_id and confirm_token reference a numbered pair from the prompt, avoiding
    the need to echo back exact entity names (which can introduce subtle variations).

    Actions:
        - If rename is None: merge child into parent from the pair
        - If rename is set: rename child to the specified standard name,
          then re-evaluate for merge opportunities in subsequent iterations
    """

    pair_id: int = Field(description="ID of the entity pair (from the prompt)")
    confirm_token: str = Field(
        min_length=4,
        max_length=4,
        description="Confirmation token from the prompt (e.g., 'xK7m')",
    )
    rename: str | None = Field(
        default=None,
        description=(
            "New standard name if renaming (e.g., 'TGF-β', 'PPARγ'); "
            "None to merge into parent without renaming"
        ),
    )
    reasoning: str = Field(
        min_length=20,
        description="Explanation: why merge or what the rename accomplishes",
    )

    @model_validator(mode="after")
    def validate_rename_not_empty(self) -> "EntityConsolidationDecision":
        """Ensure rename is non-empty if provided."""
        if self.rename is not None and not self.rename.strip():
            raise ValueError(
                "rename must be non-empty string if provided (use None to merge)"
            )
        return self


class EntityConsolidationDecisions(BaseModel):
    """LLM output: batch of entity consolidation decisions.

    Only includes pairs that should merge or be renamed. Pairs not included
    are implicitly skipped (kept separate).
    """

    decisions: list[EntityConsolidationDecision] = Field(
        description="Pairs to merge or rename (omit pairs that should remain separate)"
    )


class ClusterDecision(BaseModel):
    """LLM judgment on whether a cluster should merge, split, or exclude a member.

    The LLM makes a simple semantic judgment about cluster quality.
    Python code handles the algorithmic details using the merge tree.
    """

    group_id: str = Field(description="Cluster ID from prompt")
    action: Literal["merge", "split", "exclude"] = Field(
        default="merge",
        description=(
            "Decision on this cluster (default: merge):\n"
            "- merge: All members represent the same entity → merge to target\n"
            "- split: Cluster mixes unrelated entities → split at weakest link\n"
            "- exclude: One specific member doesn't belong → remove it from cluster"
        ),
    )
    target: str | None = Field(
        default=None,
        description=(
            "Meaning depends on action:\n"
            "- merge: target canonical name (member number, name, or new name)\n"
            "- exclude: member to remove (member number or name)\n"
            "- split: not used (omit this field)"
        ),
    )
    reasoning: str = Field(description="Explanation for the decision")


class ClusterDecisions(BaseModel):
    """LLM output: batch of cluster quality judgments."""

    decisions: list[ClusterDecision] = Field(description="Decision for each cluster")


# =============================================================================
# Paper quality assessment models
# =============================================================================


class QualityDimensionScore(BaseModel):
    """Score and justification for a single paper quality dimension.

    Each dimension is scored 0-4 with specific anchors defined in the
    field descriptions of PaperQualityAssessment. Use the discriminating
    cues provided for each dimension to distinguish between adjacent scores.
    """

    score: Literal[0, 1, 2, 3, 4]
    justification: str = Field(
        min_length=20,
        description="Concise justification citing specific features observed in the paper",
    )


class PaperQualityAssessment(BaseModel):
    """LLM assessment of scientific paper quality for screening.

    Seven-dimension rubric based on metascience findings about what reliably
    distinguishes credible from unreliable research. Each dimension scored 0-4,
    yielding overall score 0-28.

    Scoring principles:
        - Assess scientific content only; do not reward or penalise writing quality
        - Use the discriminating cues to distinguish between adjacent scores
        - When uncertain between adjacent scores, prefer the lower score
        - Score 4 requires meeting all listed criteria explicitly
        - Perfect metrics (100% accuracy, AUC=1.0) without acknowledged limitations
          indicate potential problems, not quality

    Interpretation tiers:
        0-7:   EXCLUDE - serious deficiencies; do not use without independent verification
        8-14:  CAUTION - notable gaps; usable for hypothesis generation, verify key claims
        15-21: ACCEPTABLE - sound methodology; suitable for evidence synthesis
        22-28: HIGH_TRUST - rigorous and transparent; high confidence in findings
    """

    method_clarity: QualityDimensionScore = Field(
        description="""Methodological clarity: Could a competent researcher replicate the study?
        0: Methods absent or incoherent; cannot determine what was done
        1: Major steps missing or vague; workflow not followable
        2: Workflow clear but key parameters missing (tools named without versions, thresholds unstated)
        3: Parameters stated; gaps are minor (equipment brand, buffer lot); could replicate without contacting authors
        4: All of: tool versions, parameter values, QC criteria, and workflow order explicitly stated
        Cue (2 vs 3): Can you write the full protocol, or only name the steps?"""
    )
    data_provenance: QualityDimensionScore = Field(
        description="""Data provenance: Could you reconstruct how the final sample set was derived?
        0: Data source not stated or contradictory; sample origin unknown
        1: Source named but major uncertainty remains (selection criteria absent, sample size unstated or changes between sections)
        2: Selection criteria partially described; QC mentioned but not quantified (e.g., "outliers removed" without threshold)
        3: Selection criteria explicit; QC quantified with thresholds; sample flow traceable
        4: All of: source, selection, exclusions, processing steps, QC thresholds, and sample counts at each stage
        Cue (2 vs 3): Are QC thresholds stated as numbers, or only described in words?"""
    )
    statistical_rigour: QualityDimensionScore = Field(
        description="""Statistical rigour: Are the statistical methods appropriate and completely reported?
        0: No statistical methods; claims without quantitative support
        1: Statistics present but flawed (wrong test for data type, p-values without sample sizes, no uncertainty measures)
        2: Correct tests but incomplete reporting (p-values without effect sizes or confidence intervals; missing multiple-testing correction)
        3: Appropriate tests with effect sizes and uncertainty; correction applied where needed; omissions minor
        4: All of: correct tests, effect sizes, confidence intervals, multiple-testing correction, plus validation or sensitivity analysis
        Cue (1 vs 2): Is the test wrong for the data, or correct but underreported?
        Cue (2 vs 3): Are effect sizes and uncertainty measures present, or just p-values?"""
    )
    internal_consistency: QualityDimensionScore = Field(
        description="""Internal consistency: Do methods, results, and claims align without contradiction?
        0: Clear contradictions (sample sizes change between sections, results impossible given methods, figures contradict tables)
        1: Multiple unexplained gaps (results reported for analyses not described in methods, key numbers don't reconcile)
        2: Generally coherent but loose ends (some results lack corresponding methods, minor numerical mismatches)
        3: Consistent throughout; all results trace to described methods; discrepancies are cosmetic (rounding-level)
        4: Fully traceable: every result maps to a described method; figures, tables, and text cross-reference accurately
        Cue (2 vs 3): Are there results with no corresponding method, or just small rounding differences?"""
    )
    plausibility: QualityDimensionScore = Field(
        description="""Plausibility of claims: Are conclusions proportional to the evidence presented?
        0: Extraordinary claims without evidence ("100% accuracy", "cures disease", "definitive proof")
        1: Causal language for correlational data; conclusions claim more than results show; implausible effect sizes unremarked
        2: Claims generally supported but hedging inconsistent; abstract overstates relative to results
        3: Claims match results throughout; abstract accurately reflects findings; consistent hedging; limitations mentioned
        4: All of: specific limitations named with implications, alternative explanations considered, effect sizes contextualised, conclusions explicitly bounded
        Cue (2 vs 3): Does the abstract promise more than the results deliver?"""
    )
    reproducibility_signals: QualityDimensionScore = Field(
        description="""Reproducibility signals: Does the paper describe materials sufficient to reproduce the work?
        0: Nothing mentioned (no code, data, protocols, supplement, or accession numbers)
        1: Mentioned but restricted ("available on request", "proprietary", no repository or specifics provided)
        2: Partial materials (code without data, or protocol without key parameters); reviewable but not reproducible
        3: Key materials described with locations or specifics; reproducible with reasonable effort
        4: Fully specified: all materials, parameters, procedures, and environment/conditions described
        Cue (2 vs 3): Could you reproduce the core work from what's described, or only review the approach?"""
    )
    integrity_indicators: QualityDimensionScore = Field(
        description="""Integrity indicators: Are there red flags suggesting problems, or green flags demonstrating conscientiousness?
        0: Strong red flags (duplicated figures/data across conditions, impossible values, results implausibly consistent across replicates)
        1: Weak red flags (suspiciously perfect results like 100% accuracy or all p-values just under 0.05, generic methods lacking study-specific details)
        2: Neutral (no red flags detected, standard scientific structure, no notable positive signals)
        3: Positive signals (limitations substantively discussed, conflicts disclosed, negative or null results reported)
        4: Exemplary (preregistration stated, open peer review noted, all materials stated as open, negative results prominent)
        Cue (1 vs 2): Perfect metrics (100% sensitivity, AUC=1.0, zero false positives) without acknowledged limitations are a red flag, not a strength.
        Cue (2 vs 3): Does the paper actively demonstrate care, or merely lack obvious problems?"""
    )

    @property
    def overall_score(self) -> int:
        """Sum of all dimension scores (0-28)."""
        return sum(getattr(self, f).score for f in type(self).model_fields)


class RelationshipConsolidation(BaseModel):
    """LLM output: combined mapping and polarity classification for one relationship.

    This unified model handles both semantic consolidation (merging synonyms)
    and polarity classification (positive/negative/neutral/irrelevant) in a
    single operation, as both require understanding the relationship semantics.
    """

    original: str = Field(
        description="Original relationship label to consolidate", min_length=1
    )
    consolidated: str = Field(
        description="Canonical relationship label (may equal original)", min_length=1
    )
    polarity: Literal["positive", "negative", "neutral", "irrelevant"] = Field(
        description="Biological polarity: positive (promoting/increasing), negative (inhibiting/decreasing), neutral (ambiguous direction), or irrelevant (wrong level of analysis)"
    )
    opposites: list[str] = Field(
        default_factory=list,
        description="Relationship labels with opposite biological effects (e.g., 'inhibits' opposes 'activates'). Empty list if no clear opposites exist.",
    )
    reasoning: str = Field(
        min_length=30,
        description="Explanation of consolidation and polarity classification",
    )


class RelationshipConsolidations(BaseModel):
    """LLM output: batch of relationship consolidations with polarity."""

    consolidations: list[RelationshipConsolidation] = Field(
        description="All relationship consolidations and polarities"
    )


# =============================================================================
# Consolidation provenance models
# =============================================================================


class EntityMergeRule(BaseModel):
    """Single entity merge with provenance.

    Records how one entity name was mapped to another during consolidation,
    including the reasoning (automatic rule or LLM decision). The entity kind
    is implicit from the parent container.
    """

    source: str = Field(description="Original normalized entity name")
    target: str = Field(description="Target canonical name after consolidation")
    reasoning: str = Field(
        description="Reason for merge: 'auto:cap', 'auto:fuzzy', etc. or LLM reasoning"
    )


class EntityKindMerges(BaseModel):
    """Entity merge operations for a single entity kind.

    Separates automatic merges (rule-based) from LLM-decided merges,
    and tracks which entity clusters were presented for LLM review.
    """

    automatic: list[EntityMergeRule] = Field(
        default_factory=list,
        description="Merges from automatic rules (capitalization, fuzzy match, etc.)",
    )
    clusters: list[list[str]] = Field(
        default_factory=list,
        description="Entity clusters presented to LLM for review",
    )
    llm_decided: list[EntityMergeRule] = Field(
        default_factory=list,
        description="Merges decided by LLM after cluster review",
    )


class EntityConsolidation(BaseModel):
    """Complete entity consolidation provenance.

    Tracks entity counts before and after consolidation, plus all merge
    operations that occurred. All fields are grouped by entity kind.
    """

    initial: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Entity mention counts before consolidation: {kind: {name: count}}",
    )
    merges: dict[str, EntityKindMerges] = Field(
        default_factory=dict,
        description="Merge operations by entity kind: {kind: EntityKindMerges}",
    )
    final: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Entity mention counts after consolidation: {kind: {name: count}}",
    )


class ConsolidatedData(BaseModel):
    """Unified consolidation provenance for entities and relationships.

    Primary authoritative source for all consolidation decisions made during
    the extraction pipeline. Built incrementally as each consolidation node runs.
    """

    entities: EntityConsolidation = Field(default_factory=EntityConsolidation)
    relationships: list[RelationshipConsolidation] = Field(default_factory=list)


class ProximalPairInfo(BaseModel):
    """LLM output for a single pair extracted from a proximal region."""

    entity1: str = Field(description="First entity (canonical name)")
    entity2: str = Field(description="Second entity (canonical name)")
    relationship_types: list[str] = Field(
        description="Short (1-2 word) general relationship labels like 'regulates', 'binds', 'treats'. Must be reusable across any entity pair - no entity names or specific details.",
        min_length=1,
    )
    supporting_quotes: list[str] = Field(
        description="Direct quotes supporting this association", min_length=1
    )


class ProximalPairExtraction(BaseModel):
    """LLM output: all pairs found in a proximal region."""

    pairs: list[ProximalPairInfo] = Field(
        description="Associations found in this text region"
    )
    reasoning: str = Field(
        min_length=20, description="Brief explanation of extraction choices"
    )


class EvidenceQuality(BaseModel):
    """Assessment of evidence strength based on observable properties.

    Structured decomposition of evidence quality into observable factors,
    with a probability-anchored overall score. Used for both per-document
    and cross-document assessments.
    """

    directness: Literal["explicit", "implied", "tangential"] = Field(
        description="""How directly does the source state the relationship?
        explicit: Directly states the relationship ("X causes Y", "X regulates Y")
        implied: Relationship can be inferred but is not directly stated
        tangential: Entities mentioned but relationship is peripheral or unclear"""
    )
    source_type: Literal["primary", "review", "other"] = Field(
        description="""What type of source is this evidence from?
        primary: Original experimental or clinical research reporting new findings
        review: Systematic review, meta-analysis, or narrative review of existing work
        other: Commentary, editorial, hypothesis paper, or unclear provenance"""
    )
    specificity: Literal["mechanistic", "associative", "vague"] = Field(
        description="""How specific is the evidence about the relationship?
        mechanistic: Describes pathway, mechanism, or causal chain explaining how/why
        associative: Reports statistical association, correlation, or co-occurrence with data
        vague: General statement without specific data or mechanistic detail"""
    )
    language: Literal["definitive", "hedged", "speculative"] = Field(
        description="""How certain is the language used in the source?
        definitive: Asserted as established fact ("causes", "is required for", "we found")
        hedged: Qualified but positive ("may contribute", "is associated with", "suggests")
        speculative: Uncertain ("might", "could potentially", "remains to be determined")"""
    )
    overall: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9] = Field(
        description="""Based on the factors above, what is the probability that this
        relationship is real and accurately represented in the source?
        1 - <5%:  Evidence contradicts or argues against the relationship
        2 - ~10%: No meaningful support; entities co-occur but relationship unsupported
        3 - ~20%: Speculative only; vague language ("might", "could potentially")
        4 - ~35%: Implied or hedged; suggestive but relationship not directly stated
        5 - ~50%: Associative evidence; correlation or co-occurrence without mechanism
        6 - ~65%: Explicitly stated but qualified (hedged language OR review/secondary source)
        7 - ~80%: Explicit + clear from credible source; minor limitations only
        8 - ~90%: Explicit + specific + definitive language; strong support
        9 - >95%: Mechanistic detail from primary research with definitive language
        The level should be consistent with the factors above. When uncertain
        between adjacent levels, prefer the lower one."""
    )

    # Level descriptors for display
    _LABELS: ClassVar[dict[int, str]] = {
        1: "None",
        2: "Minimal",
        3: "Tenuous",
        4: "Weak",
        5: "Limited",
        6: "Moderate",
        7: "Good",
        8: "Strong",
        9: "Robust",
    }

    @property
    def label(self) -> str:
        """Human-readable label for the evidence level."""
        return self._LABELS.get(self.overall, "Unknown")


class PairEvidenceJudgment(BaseModel):
    """LLM assessment of pair evidence in a single document.

    Evaluates the strength of evidence and selects the most appropriate
    relationship type from candidates.

    Field order is intentional: reasoning and quote selection come first
    to encourage the model to analyze evidence before committing to
    quality assessments.
    """

    supporting_quote_ids: list[int] = Field(
        description="Indices (0-based) of quotes that support this assessment"
    )
    reasoning: str = Field(
        min_length=30,
        description="Explanation of evidence assessment and relationship choice",
    )
    relationship: str = Field(
        description="Selected relationship type (from candidates or new)"
    )
    evidence: EvidenceQuality = Field(
        description="Structured assessment of evidence quality"
    )
    topic_relevance: Literal[1, 2, 3, 4, 5] = Field(
        description="""How central is this pair to the research topic?
        1: Tangential - incidental co-occurrence only
        2: Peripheral - broadly related but not specific to the topic
        3: Related - connected via known mechanisms or associations
        4: Directly relevant - involves core aspects of the topic
        5: Central - directly addresses the research question"""
    )


class CrossDocumentJudgment(BaseModel):
    """LLM final judgment synthesizing evidence across all documents.

    Field order is intentional: reasoning comes first to encourage the model
    to think through the evidence before committing to assessments. This
    improves calibration of confidence estimates (Becker & Soatto, 2024).
    """

    reasoning: str = Field(
        min_length=50, description="Detailed explanation of the decision"
    )
    relationship: str = Field(
        description="Selected final relationship type (most accurate overall)"
    )
    evidence: EvidenceQuality = Field(
        description="Synthesized evidence quality across all documents"
    )
    accepted: bool = Field(
        description="Whether to accept this pair as a valid association"
    )
    decision_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability that this accept/reject decision is correct (0.0-1.0)",
    )
    topic_relevance: Literal[1, 2, 3, 4, 5] = Field(
        description="""How central is this pair to the research topic?
        1: Tangential - incidental co-occurrence only
        2: Peripheral - broadly related but not specific to the topic
        3: Related - connected via known mechanisms or associations
        4: Directly relevant - involves core aspects of the topic
        5: Central - directly addresses the research question"""
    )


# =============================================================================
# Final pipeline output models
# =============================================================================


class SimpleEntity(BaseModel):
    """Lightweight entity representation for final output.

    Contains only the essential information needed to identify an entity
    without the full provenance chain.
    """

    name: str = Field(description="Canonical entity name")
    kind: str = Field(description='Entity type (e.g., "gene", "disease")')
    aliases: list[str] = Field(description="Alternative names found in text")


class PairAssessment(BaseModel):
    """Evidence assessment for one pair in one document.

    This is the core result of per-document pair analysis, containing
    the selected relationship, evidence quality, and supporting evidence.

    Attributes:
        resource_id: Which document this assessment is from
        entity1: First entity reference (canonical name + original mentions)
        entity2: Second entity reference (canonical name + original mentions)
        relationship: Selected relationship type
        quotes: Supporting quotes from this document
        evidence: Structured assessment of evidence quality
        reasoning: Explanation of evidence assessment
        source: How this assessment was discovered ("direct" from initial
            document extraction, "sweep" from co-mention sweep pass)
    """

    resource_id: ResourceId
    entity1: EntityRef
    entity2: EntityRef
    relationship: str
    quotes: list[ResourceQuote]
    evidence: EvidenceQuality
    topic_relevance: Literal[1, 2, 3, 4, 5]
    reasoning: str
    source: Literal["direct", "sweep"] = "direct"


class PairSpread(BaseModel):
    """Assessments grouped by relationship polarity.

    Organizes per-document assessments by the biological polarity of their
    relationship labels. Polarity is derived via lookup from relationship label
    to polarity mapping created during consolidation.

    This structure enables explicit synthesis of positive vs negative evidence
    and identification of contentious pairs where biological effects contradict.

    Attributes:
        positive: Assessments with promoting/increasing/activating relationships
        negative: Assessments with inhibiting/decreasing/protective relationships
        neutral: Assessments with ambiguous directionality
        irrelevant: Assessments with relationships orthogonal to research question
    """

    positive: list[PairAssessment] = Field(default_factory=list)
    negative: list[PairAssessment] = Field(default_factory=list)
    neutral: list[PairAssessment] = Field(default_factory=list)
    irrelevant: list[PairAssessment] = Field(default_factory=list)


class PairJudgment(BaseModel):
    """Final cross-document judgment on an entity pair.

    This is the primary output of the extraction pipeline, containing
    all per-document assessments organized by polarity and the final
    accept/reject decision.

    The final polarity is inferred from the selected relationship via the
    polarity mapping created during consolidation.
    """

    entity1: SimpleEntity = Field(description="First entity")
    entity2: SimpleEntity = Field(description="Second entity")
    relationship: str = Field(description="Final relationship type")
    spread: PairSpread = Field(
        description="Per-document assessments grouped by relationship polarity"
    )
    accepted: bool = Field(description="Whether this pair is accepted")
    evidence: EvidenceQuality = Field(
        description="Synthesized evidence quality for final judgment"
    )
    topic_relevance: Literal[1, 2, 3, 4, 5] = Field(
        description="How central this pair is to the research topic (1-5)"
    )
    decision_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability that this accept/reject decision is correct (0.0-1.0)",
    )
    reasoning: str = Field(description="Explanation of final decision")

    def iter_assessments(self) -> Iterable[PairAssessment]:
        """Iterate over all assessments regardless of polarity.

        Provides a compatibility bridge for legacy code that expected a flat
        ``assessments`` list on ``PairJudgment``. Prefer accessing the
        structured ``spread`` attribute directly when possible.
        """

        yield from self.spread.positive
        yield from self.spread.negative
        yield from self.spread.neutral
        yield from self.spread.irrelevant

    @property
    def assessments(self) -> list[PairAssessment]:
        """Backward-compatible view exposing all assessments as a list."""

        return list(self.iter_assessments())


class ExtractionMetadata(BaseModel):
    """Summary statistics for extraction run.

    When extraction is incomplete, resume_from and resume_state contain
    checkpoint information for resumption. Both are None when complete.
    """

    topic: str
    resource_count: int
    total_entities_found: int
    entities_after_validation: int
    entities_merged: int
    merge_cache_hits: int
    merge_cache_misses: int
    proximal_sets_found: int
    total_pairs_found: int
    pairs_accepted: int
    pairs_rejected: int
    quotes_validated: int
    quotes_failed: int
    # Resumption fields (both None when extraction complete)
    resume_from: (
        Literal[
            "process_documents",
            "consolidate_entities",
            "consolidate_relationships",
            "sweep_co_mentions",
            "consolidate_new_relationships",
            "consolidate_by_neighbours",
        ]
        | None
    ) = Field(None, description="Last completed stage (None if complete)")
    resume_state: dict | None = Field(
        None, description="Serialized State for resumption"
    )

    @property
    def is_complete(self) -> bool:
        """True if extraction completed successfully."""
        return self.resume_from is None

    @property
    def is_resumable(self) -> bool:
        """True if extraction can be resumed from a checkpoint."""
        return self.resume_from is not None and self.resume_state is not None


class ExtractionResult(BaseModel):
    """Final pipeline output.

    Contains accepted pair judgments with full provenance and shared resource pool.
    Provides efficient serialization by storing resources once and referencing
    by ID in quotes.
    """

    topic: str = Field(description="Research topic for extraction context")
    target_entity_types: list[str] = Field(
        description="Entity types that were extracted"
    )
    permitted_pairs: dict[str, list[str]] = Field(
        description="Permitted entity kind pairs for filtering"
    )
    resources: ResourcePool = Field(
        description="Shared pool of all resources referenced by quotes"
    )
    judgments: list[PairJudgment] = Field(
        description="All pair judgments (accepted and rejected)"
    )
    metadata: ExtractionMetadata = Field(description="Extraction statistics")
    consolidated: ConsolidatedData = Field(
        default_factory=ConsolidatedData,
        description="Unified consolidation provenance for entities and relationships",
    )
    entities: dict[str, EntityRef] = Field(
        default_factory=dict,
        description="Global entity index with aggregated mentions from all resources",
    )
    paper_quality: dict[str, PaperQualityAssessment] = Field(
        default_factory=dict,
        description="Paper quality assessments keyed by resource URL",
    )

    @model_serializer(mode="wrap")
    def _serialize(self, serializer, info):
        """Serialize with full EntityRef data in entities dict."""
        if info.mode != "json":
            return serializer(self)
        # Do default serialization (EntityRefs become strings via their serializer)
        data = serializer(self)
        # Manually serialize entities dict with full EntityRef data
        data["entities"] = {
            canonical: {
                "canonical": ref.canonical,
                "mentions": to_jsonable_python(
                    ref.mentions, fallback=lambda x: x.model_dump(mode="json")
                ),
            }
            for canonical, ref in self.entities.items()
        }
        return data

    @model_validator(mode="before")
    @classmethod
    def _rehydrate_quotes(cls, data):
        """Restore Resource and EntityRef objects from URL/canonical references."""
        from interaction_finder.checkpoint import (
            _needs_rehydration,
            _rehydrate_judgments_quotes,
        )

        if not isinstance(data, dict) or "resources" not in data:
            return data
        judgments = data.get("judgments", [])
        if not _needs_rehydration(judgments):
            return data
        pool = ResourcePool.model_validate(data["resources"])
        entities = data.get("entities")
        _rehydrate_judgments_quotes(judgments, pool, entities)
        return data
