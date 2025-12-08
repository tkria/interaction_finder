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
from typing import Iterable, Literal, NamedTuple

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
    mentions: list[EntityMention]

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
    """LLM output: all entities found in document."""

    entities: list[EntityInfo] = Field(description="Entities found in document")


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


class PairEvidenceJudgment(BaseModel):
    """LLM assessment of pair evidence in a single document.

    Evaluates the strength of evidence and selects the most appropriate
    relationship type from candidates.
    """

    relationship: str = Field(
        description="Selected relationship type (from candidates or new)"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in this pair's validity based on evidence"
    )
    reasoning: str = Field(
        min_length=30, description="Explanation of confidence and relationship choice"
    )
    supporting_quote_ids: list[int] = Field(
        description="Indices (0-based) of quotes that support this assessment"
    )


class CrossDocumentJudgment(BaseModel):
    """LLM final judgment synthesizing evidence across all documents."""

    accepted: bool = Field(
        description="Whether to accept this pair as a valid association"
    )
    relationship: str = Field(
        description="Selected final relationship type (most accurate overall)"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence level in this judgment"
    )
    reasoning: str = Field(
        min_length=50, description="Detailed explanation of the decision"
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
    the selected relationship, confidence level, and supporting evidence.

    Attributes:
        resource_id: Which document this assessment is from
        entity1: First entity reference (canonical name + original mentions)
        entity2: Second entity reference (canonical name + original mentions)
        relationship: Selected relationship type
        quotes: Supporting quotes from this document
        confidence: Qualitative confidence in this association
        reasoning: Explanation of confidence level
        source: How this assessment was discovered ("direct" from initial
            document extraction, "sweep" from co-mention sweep pass)
    """

    resource_id: ResourceId
    entity1: EntityRef
    entity2: EntityRef
    relationship: str
    quotes: list[ResourceQuote]
    confidence: Literal["high", "medium", "low"]
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
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence in final judgment"
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


class ClusteringMetadata(BaseModel):
    """Metadata about entity clustering for a specific entity kind.

    Captures hierarchical clustering results including token specificity
    weights, cluster composition, and merge tree structure for debugging
    and analysis.
    """

    kind: str = Field(description="Entity kind (e.g., 'gene', 'phenotype')")
    total_entities: int = Field(
        description="Total entities of this kind before clustering"
    )
    entities_in_clustering: int = Field(
        description="Entities that participated in clustering (after auto-merge)"
    )
    threshold: float = Field(description="Similarity threshold used for clustering")

    # Token specificity weights (IDF-like, for debugging)
    token_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Token -> specificity score (log-scaled IDF)",
    )
    weight_range: tuple[float, float] = Field(
        description="(min_weight, max_weight) across all tokens"
    )

    # Clustering results
    clusters_formed: int = Field(description="Number of clusters produced")
    largest_cluster_size: int = Field(
        description="Size of largest cluster (for detecting hierarchies)"
    )
    multi_entity_clusters: int = Field(
        description="Clusters with 2+ entities (presented to LLM)"
    )
    singleton_clusters: int = Field(
        description="Clusters with 1 entity (kept separate)"
    )

    # Cluster composition (for analysis)
    cluster_sizes: list[int] = Field(
        description="Size of each cluster, sorted descending"
    )
    large_clusters: list[list[str]] = Field(
        default_factory=list,
        description="Entity names in clusters with 5+ members (hierarchies)",
    )

    # Hierarchical cluster structure (merge trees)
    merge_trees: list[dict] = Field(
        default_factory=list,
        description="Hierarchical cluster trees showing merge structure and similarities",
    )

    # Sample token weights for common terms (debugging aid)
    sample_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Specificity scores for frequent tokens (up to 10)",
    )


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

    # Clustering metadata (per entity kind)
    clustering_metadata: dict[str, ClusteringMetadata] = Field(
        default_factory=dict,
        description="Clustering statistics by entity kind for debugging",
    )

    # Resumption fields (both None when extraction complete)
    resume_from: (
        Literal[
            "process_documents",
            "consolidate_entities",
            "consolidate_relationships",
            "sweep_co_mentions",
            "consolidate_new_relationships",
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


class ConsolidationRule(BaseModel):
    """Single entity consolidation rule for provenance tracking.

    Records how one entity name was mapped to another during consolidation.
    """

    source: str = Field(description="Original normalized entity name")
    kind: str = Field(description="Entity kind (e.g., 'gene', 'phenotype')")
    target: str = Field(description="Target canonical name after consolidation")
    reasoning: str = Field(
        description="Reason for consolidation: 'auto:cap', 'auto:fuzzy', 'cached', or LLM reasoning"
    )


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
    consolidation_rules: list[ConsolidationRule] = Field(
        default_factory=list,
        description="Entity consolidation rules applied during extraction",
    )
    relationship_oppositions: dict[str, set[str]] = Field(
        default_factory=dict,
        description="Mapping of relationships to their semantic opposites",
    )

    @model_serializer(mode="wrap")
    def _serialize(self, serializer, info):
        """Collect unique EntityRefs and serialize with entities dict."""
        if info.mode != "json":
            return serializer(self)

        # Collect unique EntityRefs before serialization
        entities: dict[str, EntityRef] = {}
        for judgment in self.judgments:
            for assessment in judgment.iter_assessments():
                if assessment.entity1.canonical not in entities:
                    entities[assessment.entity1.canonical] = assessment.entity1
                if assessment.entity2.canonical not in entities:
                    entities[assessment.entity2.canonical] = assessment.entity2

        # Do default serialization (EntityRefs become strings via their serializer)
        data = serializer(self)

        # Manually serialize entities dict with full EntityRef data
        # We serialize mentions in JSON mode (to convert ResourceQuote.resource to resource_url)
        # but keep the top-level EntityRef structure (canonical + mentions)
        data["entities"] = {
            canonical: {
                "canonical": ref.canonical,
                "mentions": to_jsonable_python(
                    ref.mentions, fallback=lambda x: x.model_dump(mode="json")
                ),
            }
            for canonical, ref in entities.items()
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
