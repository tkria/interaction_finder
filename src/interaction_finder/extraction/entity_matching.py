"""Entity matching with composable speculation system.

This module provides unified entity matching for both:
1. Pair extraction / quote validation (finding which entity a name refers to)
2. Entity consolidation (grouping entities that should merge)

Key concepts:
- Variant extraction: Each entity expands to multiple variant forms
- Speculation levels: Track confidence in each variant (0 = most confident)
- Composable speculation: variant_spec + match_penalty = total_speculation
- Contested variants: Forms mapping to multiple entities are excluded
- Auto-merge threshold: Only high-confidence matches merge automatically

Speculation system:
- 0: Original canonical/alias form
- 1: Before parenthesis (e.g., "ACTB" from "ACTB (β-Actin)")
- 2: Primary content (main expansion or alternation)
- 3: Secondary content
- 4: Nested expansion (alternations within parentheses)
- +0: Exact match penalty
- +3: Fuzzy match penalty (plurals, UK/US spelling)
- +1: Alias offset (aliases less confident than canonical)

AUTO_MERGE_THRESHOLD = 4: Auto-merges spec ≤ 4, agent reviews spec > 4
"""

from dataclasses import dataclass
from typing import Literal
import re

from interaction_finder.extraction.utils import (
    normalize_for_comparison,
    _expand_slash,
    _is_valid_entity_form,
    is_obvious_variant,
    osa_distance,
    entity_names_match,
)


# ============================================================================
# Speculation Constants
# ============================================================================

# Variant speculation levels (from extraction)
SPEC_ORIGINAL = 0  # Unmodified canonical/alias
SPEC_BEFORE_PAREN = 1  # "ACTB" from "ACTB (β-Actin)"
SPEC_PRIMARY_CONTENT = 2  # Main content (expansion/alternation)
SPEC_SECONDARY_CONTENT = 3  # Secondary content
SPEC_NESTED_EXPANSION = 4  # Nested alternations

# ============================================================================
# Token Filtering for Clustering
# ============================================================================

# English stopwords (NLTK list) - filtered from token overlap clustering
# to avoid spurious matches on common words like "the", "and", etc.
_STOPWORDS = frozenset(
    {
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "ourselves",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "he",
        "him",
        "his",
        "himself",
        "she",
        "her",
        "hers",
        "herself",
        "it",
        "its",
        "itself",
        "they",
        "them",
        "their",
        "theirs",
        "themselves",
        "what",
        "which",
        "who",
        "whom",
        "this",
        "that",
        "these",
        "those",
        "am",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "having",
        "do",
        "does",
        "did",
        "doing",
        "a",
        "an",
        "the",
        "and",
        "but",
        "if",
        "or",
        "because",
        "as",
        "until",
        "while",
        "of",
        "at",
        "by",
        "for",
        "with",
        "about",
        "against",
        "between",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "to",
        "from",
        "up",
        "down",
        "in",
        "out",
        "on",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "s",
        "t",
        "can",
        "will",
        "just",
        "don",
        "should",
        "now",
    }
)

# Match quality penalties (added to variant speculation)
PENALTY_EXACT = 0  # Exact normalized match
PENALTY_FUZZY = 3  # Fuzzy match (validated)

# Alias offset (added during extraction for alias-sourced variants)
OFFSET_ALIAS = 1

# Consolidation threshold
AUTO_MERGE_THRESHOLD = 4  # Total speculation ≤ 4 → auto-merge
# Total speculation > 4 → agent review

# Fuzzy matching thresholds
MIN_LENGTH_FOR_FUZZY = 10  # Only fuzzy-match entities with ≥10 chars
MIN_FUZZY_SIMILARITY = 0.7  # 70% minimum similarity for fuzzy matches


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class SpeculatedVariant:
    """Variant form with speculation level and provenance."""

    form: str
    speculation: int  # 0 = most confident
    source: str  # original, before_paren, paren_expansion, etc.
    is_from_alias: bool = False

    def __repr__(self) -> str:
        alias_marker = " [alias]" if self.is_from_alias else ""
        return f"[{self.speculation}] {self.form}{alias_marker} ({self.source})"


@dataclass
class EntityMatch:
    """Match result with full speculation tracking."""

    canonical: str
    matched_variant: SpeculatedVariant
    query_variant: SpeculatedVariant
    match_penalty: int  # PENALTY_EXACT or PENALTY_FUZZY

    @property
    def total_speculation(self) -> int:
        return self.matched_variant.speculation + self.match_penalty

    def auto_trigger(self) -> str:
        """Format: <total>:<source>:<match_kind>"""
        source = self.matched_variant.source
        if self.matched_variant.is_from_alias:
            source = f"alias:{source}"
        match_kind = "exact" if self.match_penalty == PENALTY_EXACT else "fuzzy"
        return f"{self.total_speculation}:{source}:{match_kind}"


@dataclass
class VariantMapping:
    """Maps normalized variant to canonical with speculation."""

    canonical: str
    variant: SpeculatedVariant


@dataclass
class ConsolidationCandidates:
    """Entity pairs grouped by decision type."""

    auto_merge: list[tuple[str, str, str]]  # (child, parent, reasoning)
    agent_review: list[tuple[str, str]]  # (child, parent)
    agent_review_groups: list[
        frozenset[str]
    ]  # Groups of 2+ related entities for batch review
    contested_warnings: list[
        tuple[str, dict[str, list[str]]]
    ]  # (normalized_form, {canonical: [variant_forms]})
    merge_trees: list  # List of Cluster trees for surgical splitting (type hint avoided for circular import)
    clustering_info: dict | None = None  # Metadata for debugging (optional)


# ============================================================================
# Parenthetical Classification
# ============================================================================


def classify_parenthetical(
    base: str, paren_content: str
) -> Literal["expansion", "qualifier"]:
    """Classify parenthetical as expansion (main content) or qualifier (supplementary).

    Returns "expansion" for multi-word, technical notation, or longer-than-base content.
    Returns "qualifier" for single lowercase words like "gene", "protein".
    """
    paren = paren_content.strip()
    # Single lowercase word without hyphens
    if paren.islower() and " " not in paren and "-" not in paren:
        return "qualifier"
    # Multi-word or technical notation (hyphens, Greek letters)
    if (
        " " in paren
        or "-" in paren
        or any(c in paren for c in "αβγδεζηθικλμνξοπρστυφχψω")
    ):
        return "expansion"
    # Longer than base
    if len(paren) > len(base):
        return "expansion"
    return "qualifier"


# ============================================================================
# Variant Extraction
# ============================================================================


def _generate_acronyms(text: str) -> list[str]:
    """Generate nested and direct acronyms from entity name.

    Generates both:
    - Direct acronym (one letter per word)
    - Expanded acronym (expanding all-caps words into individual letters)

    Only generates if:
    - 2+ words total
    - Result not identical to original

    Returns:
        List of generated acronyms (may be empty)

    Examples:
        >>> _generate_acronyms("Idiopathic PAH")
        ['IP', 'IPAH']
        >>> _generate_acronyms("Pulmonary Arterial Hypertension")
        ['PAH']
        >>> _generate_acronyms("PAH")
        []
    """
    words = [w for w in re.split(r"[\s\-]+", text.strip()) if w]
    if len(words) < 2:
        return []

    norm_text = normalize_for_comparison(text)

    def letters_from_words(expand_acronyms: bool) -> list[str]:
        """Extract letters: expand all-caps words if requested."""
        letters = []
        for word in words:
            if expand_acronyms and word.isupper() and len(word) > 1:
                letters.extend(word)
            elif word[0].isupper():
                letters.append(word[0])
        return letters

    # Generate both expanded (nested) and direct acronyms
    # Nested comes first (preserves all-caps words in entirety)
    candidates = [
        "".join(letters_from_words(expand_acronyms=True)),  # Expanded/Nested
        "".join(letters_from_words(expand_acronyms=False)),  # Direct
    ]

    # Filter: unique, different from original, at least 2 letters
    return [
        acronym
        for acronym in dict.fromkeys(candidates)
        if len(acronym) >= 2 and normalize_for_comparison(acronym) != norm_text
    ]


def extract_entity_variants(
    entity_name: str,
    aliases: list[str] | None = None,
    _is_alias: bool = False,
) -> list[SpeculatedVariant]:
    """Extract all variant forms with speculation levels.

    May return duplicates with different speculation levels. Caller should deduplicate
    if needed (build_variant_map handles this). Aliases processed recursively with
    +OFFSET_ALIAS speculation offset.
    """
    variants: list[SpeculatedVariant] = []
    base_offset = OFFSET_ALIAS if _is_alias else 0

    # Always include original
    variants.append(
        SpeculatedVariant(
            form=entity_name,
            speculation=SPEC_ORIGINAL + base_offset,
            source="original",
            is_from_alias=_is_alias,
        )
    )

    # Check for parenthetical: "Base (Content)"
    paren_match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", entity_name.strip())

    if paren_match:
        base = paren_match.group(1).strip()
        paren_content = paren_match.group(2).strip()

        # Add base (before parenthesis)
        if base and _is_valid_entity_form(base):
            variants.append(
                SpeculatedVariant(
                    form=base,
                    speculation=SPEC_BEFORE_PAREN + base_offset,
                    source="before_paren",
                    is_from_alias=_is_alias,
                )
            )

        # Assign speculation based on whether paren is expansion or qualifier
        paren_type = classify_parenthetical(base, paren_content)
        if paren_type == "expansion":
            # Paren is primary, base alts secondary, paren alts nested
            paren_spec = SPEC_PRIMARY_CONTENT
            base_alt_spec = SPEC_SECONDARY_CONTENT
            paren_alt_spec = SPEC_NESTED_EXPANSION
        else:
            # Base alts primary, paren secondary, paren alts nested
            paren_spec = SPEC_SECONDARY_CONTENT
            base_alt_spec = SPEC_PRIMARY_CONTENT
            paren_alt_spec = SPEC_NESTED_EXPANSION

        # Determine source labels based on type
        paren_source = (
            "paren_expansion" if paren_type == "expansion" else "paren_qualifier"
        )

        # Extract alternations
        base_alternations = _expand_slash(base)
        base_has_slash = len(base_alternations) > 1
        paren_alternations = _expand_slash(paren_content)
        paren_has_slash = len(paren_alternations) > 1

        # Add parenthetical content (always add if valid, regardless of slashes)
        if _is_valid_entity_form(paren_content):
            variants.append(
                SpeculatedVariant(
                    form=paren_content,
                    speculation=paren_spec + base_offset,
                    source=paren_source,
                    is_from_alias=_is_alias,
                )
            )

        # Add base alternations (if has slash)
        if base_has_slash:
            for alt in base_alternations[1:]:  # Skip original
                if _is_valid_entity_form(alt):
                    variants.append(
                        SpeculatedVariant(
                            form=alt,
                            speculation=base_alt_spec + base_offset,
                            source="base_alternation",
                            is_from_alias=_is_alias,
                        )
                    )

        # Add paren alternations (if has slash)
        if paren_has_slash:
            for alt in paren_alternations[1:]:  # Skip original
                if _is_valid_entity_form(alt):
                    variants.append(
                        SpeculatedVariant(
                            form=alt,
                            speculation=paren_alt_spec + base_offset,
                            source="paren_alternation",
                            is_from_alias=_is_alias,
                        )
                    )

    else:
        # No parentheses - check for alternations
        alternations = _expand_slash(entity_name)
        if len(alternations) > 1:
            for alt in alternations[1:]:  # Skip original (already added)
                if _is_valid_entity_form(alt):
                    variants.append(
                        SpeculatedVariant(
                            form=alt,
                            speculation=SPEC_PRIMARY_CONTENT + base_offset,
                            source="alternation",
                            is_from_alias=_is_alias,
                        )
                    )

    # Process aliases recursively (with +OFFSET_ALIAS)
    if aliases:
        for alias in aliases:
            alias_variants = extract_entity_variants(
                alias, aliases=None, _is_alias=True
            )
            variants.extend(alias_variants)

    # Generate acronym variants (only for non-aliases)
    if not _is_alias:
        acronyms = _generate_acronyms(entity_name)
        if acronyms:
            existing_norms = {normalize_for_comparison(v.form) for v in variants}

            for acronym in acronyms:
                acronym_norm = normalize_for_comparison(acronym)
                if acronym_norm not in existing_norms:
                    variants.append(
                        SpeculatedVariant(
                            form=acronym,
                            speculation=SPEC_BEFORE_PAREN + base_offset,  # Level 1
                            source="generated_acronym",
                            is_from_alias=_is_alias,
                        )
                    )

    # Sort by speculation (stable sort preserves order within level)
    variants.sort(key=lambda v: v.speculation)

    return variants


# ============================================================================
# Validation
# ============================================================================


def are_safe_capitalization_variants(names: set[str]) -> bool:
    """Check if names are safe to auto-merge (capitalization or obvious spelling variants).

    Returns True if all names in the set are variants of each other through:
    - Pure capitalization differences (BRCA1 vs Brca1)
    - Obvious spelling variants (haemorrhagic vs hemorrhagic)
    - Or both

    Rejects unsafe cases like number-only differences (SMAD6 vs SMAD7).
    """
    if len(names) <= 1:
        return True

    names_list = list(names)

    # Check all pairs match using core matching logic (includes all safety checks)
    for i, n1 in enumerate(names_list):
        for n2 in names_list[i + 1 :]:
            matched, _ = entity_names_match(n1, n2)
            if not matched:
                return False

    return True


# ============================================================================
# Variant Map Building
# ============================================================================


def build_variant_map(
    entities: dict[str, list[SpeculatedVariant]],
) -> tuple[dict[str, VariantMapping], set[str]]:
    """Build variant→canonical map, excluding contested (multi-canonical) variants."""
    # Collect all mappings
    norm_to_mappings: dict[str, list[tuple[str, SpeculatedVariant]]] = {}

    for canonical, variants in entities.items():
        for sv in variants:
            norm = normalize_for_comparison(sv.form)
            norm_to_mappings.setdefault(norm, []).append((canonical, sv))

    # Filter contested, keep best uncontested
    variant_map: dict[str, VariantMapping] = {}
    contested: set[str] = set()

    for norm, mappings in norm_to_mappings.items():
        unique_canonicals = {canonical for canonical, _ in mappings}

        if len(unique_canonicals) == 1:
            # Uncontested - take lowest speculation
            canonical, best_variant = min(mappings, key=lambda m: m[1].speculation)
            variant_map[norm] = VariantMapping(
                canonical=canonical, variant=best_variant
            )
        else:
            # Contested - exclude from map
            contested.add(norm)

    return variant_map, contested


# ============================================================================
# Entity Matching (for pair extraction / quote validation)
# ============================================================================


def find_entity_match(
    query: str,
    entities: dict[str, list[SpeculatedVariant]],
    *,
    allow_fuzzy: bool = True,
) -> EntityMatch | None:
    """Find best entity match via progressive matching (exact, then fuzzy if enabled)."""
    # Build variant map (excludes contested)
    variant_map, _ = build_variant_map(entities)

    # Augment variant map with direct canonical mappings
    # This handles contested variants: if query matches a canonical exactly,
    # it should match even if that canonical's variants are contested
    for canonical, variants in entities.items():
        canonical_norm = normalize_for_comparison(canonical)
        # Only add if not already present (avoid overwriting better matches)
        if canonical_norm not in variant_map:
            # Use the first variant (SPEC_ORIGINAL) from this entity
            original_variant = next(
                (v for v in variants if v.speculation == SPEC_ORIGINAL), variants[0]
            )
            variant_map[canonical_norm] = VariantMapping(
                canonical=canonical, variant=original_variant
            )

    # Extract query variants
    query_variants = extract_entity_variants(query)

    # Stage 1: Try exact matches (includes direct canonical matches now)
    best_match: EntityMatch | None = None

    for qv in query_variants:
        norm = normalize_for_comparison(qv.form)
        if norm in variant_map:
            mapping = variant_map[norm]
            match = EntityMatch(
                canonical=mapping.canonical,
                matched_variant=mapping.variant,
                query_variant=qv,
                match_penalty=PENALTY_EXACT,
            )
            # Keep match with lowest total speculation
            if (
                best_match is None
                or match.total_speculation < best_match.total_speculation
            ):
                best_match = match

    if best_match is not None:
        return best_match

    if not allow_fuzzy:
        return None

    # Stage 2: Fuzzy matching with validation
    scores: list[tuple[float, int, VariantMapping, SpeculatedVariant]] = []

    for qv in query_variants:
        qv_norm = normalize_for_comparison(qv.form)
        for variant_norm, mapping in variant_map.items():
            shorter_len = min(len(qv_norm), len(variant_norm))
            dist = osa_distance(qv_norm, variant_norm)
            similarity = 1 - dist / shorter_len

            # For obvious variants (spelling/hyphenation), allow shorter strings
            # Otherwise require minimum length for fuzzy matching
            if not is_obvious_variant(qv_norm, variant_norm):
                if shorter_len < MIN_LENGTH_FOR_FUZZY:
                    continue

            # Calculate length-scaled maximum distance
            max_dist = 1 + shorter_len // 10

            # Accept if within fuzzy threshold
            if dist <= max_dist and similarity >= MIN_FUZZY_SIMILARITY:
                # For dist==1, require obvious variant validation
                if dist == 1 and not is_obvious_variant(qv_norm, variant_norm):
                    continue
                scores.append((similarity, dist, mapping, qv))

    if not scores:
        return None

    # Sort by similarity (highest first)
    scores.sort(reverse=True, key=lambda x: x[0])
    best_similarity, best_dist, best_mapping, best_qv = scores[0]

    # Validation checks
    shorter_len = min(
        len(normalize_for_comparison(best_qv.form)),
        len(normalize_for_comparison(best_mapping.variant.form)),
    )

    # Minimum similarity threshold
    if best_similarity < MIN_FUZZY_SIMILARITY:
        return None

    # Maximum distance (scales with length)
    max_distance = 1 + shorter_len // 10
    if best_dist > max_distance:
        return None

    # Specificity check (gap to second-best)
    min_gap = 2 * (1 - best_similarity)
    if len(scores) > 1 and best_similarity - scores[1][0] < min_gap:
        return None

    return EntityMatch(
        canonical=best_mapping.canonical,
        matched_variant=best_mapping.variant,
        query_variant=best_qv,
        match_penalty=PENALTY_FUZZY,
    )


# ============================================================================
# Consolidation (for entity merging)
# ============================================================================


def _has_token_overlap(
    canonical1: str,
    canonical2: str,
    entities: dict[str, list[SpeculatedVariant]],
) -> bool:
    """Check if entities share tokens or if tokens match variants.

    Detects patterns like:
    1. "Idiopathic PAH" where token "PAH" matches a variant of "Pulmonary Arterial Hypertension"
    2. "Heritable pulmonary arterial hypertension" vs "Pulmonary arterial hypertension" (shared tokens)

    Requires proportional overlap: at least 50% (rounded down, minimum 1) of the
    smaller entity's token count must overlap.

    Returns:
        True if sufficient token overlap detected
    """

    def get_norms_and_tokens(canonical: str) -> tuple[set[str], set[str]]:
        """Extract normalized variant forms and tokens for an entity."""
        norms = {normalize_for_comparison(v.form) for v in entities.get(canonical, [])}
        tokens = {
            normalize_for_comparison(t)
            for t in re.split(r"[\s\-]+", canonical)
            if len(t) >= 3
        }
        return norms, tokens

    norms1, tokens1 = get_norms_and_tokens(canonical1)
    norms2, tokens2 = get_norms_and_tokens(canonical2)

    # Calculate overlap for each check type
    variant_overlap = len(tokens1 & norms2) + len(tokens2 & norms1)
    token_overlap = len(tokens1 & tokens2)
    total_overlap = max(variant_overlap, token_overlap)

    # Require at least 50% (rounded down, min 1) of smaller token count
    min_token_count = min(len(tokens1), len(tokens2))
    required_overlap = max(1, min_token_count // 2)

    return total_overlap >= required_overlap


def select_best_form(names: set[str]) -> str:
    """Select best form from equivalent names: prefer mixed-case over all-upper/all-lower."""
    return max(
        names,
        key=lambda v: (
            sum(1 for c in v if c.islower())
            * sum(1 for c in v if c.isupper()),  # Mixed
            sum(1 for c in v if c.isupper()),  # Uppercase count
            -sum(1 for c in v if c.islower()),  # Minimize lowercase
        ),
    )


def find_consolidation_candidates(
    entities: dict[str, list[SpeculatedVariant]],
    cluster_threshold: float = 0.50,
    mention_counts: dict[str, int] | None = None,
    logger=None,
) -> ConsolidationCandidates:
    """Find entity pairs for consolidation (auto-merge if ≤ threshold, else agent review).

    Args:
        entities: Dict of canonical_name → list of variant forms
        cluster_threshold: Token overlap proportion for clustering (0.0-1.0)
        mention_counts: Optional dict of entity_name → mention count for IDF weighting
        logger: Optional logger for INFO-level merge decisions
    """
    # Build variant map (excludes contested)
    variant_map, contested = build_variant_map(entities)

    # Track results
    auto_merge: list[tuple[str, str, str]] = []
    agent_review: list[tuple[str, str]] = []
    contested_warnings: list[tuple[str, dict[str, list[str]]]] = []

    # Collect merge decisions for batched logging
    cap_merges: list[tuple[str, str]] = []
    variant_merges: list[tuple[str, str]] = []
    fuzzy_merges: list[tuple[str, str]] = []

    # Phase 1: Handle contested variants
    # Check if they're safe capitalization variants. If safe, auto-merge.
    # If not, add to warnings (will be excluded from auto-merge).
    for norm in contested:
        # Build mapping of canonical -> variant forms for this contested norm
        canonical_to_variants: dict[str, list[str]] = {}
        for canonical, variants in entities.items():
            matching_variants = [
                v.form for v in variants if normalize_for_comparison(v.form) == norm
            ]
            if matching_variants:
                canonical_to_variants[canonical] = matching_variants

        if len(canonical_to_variants) <= 1:
            continue

        affected = set(canonical_to_variants.keys())

        # Check if these are safe capitalization variants (BRCA1/Brca1/brca1)
        if are_safe_capitalization_variants(affected):
            # Pick best canonical (prefer mixed-case)
            best_canonical = select_best_form(affected)

            # Create auto-merge rules for all others
            # Find the lowest speculation variant from the best canonical
            best_variants = [
                v
                for v in entities[best_canonical]
                if normalize_for_comparison(v.form) == norm
            ]
            if best_variants:
                best_variant = min(best_variants, key=lambda v: v.speculation)
                match = EntityMatch(
                    canonical=best_canonical,
                    matched_variant=best_variant,
                    query_variant=best_variant,
                    match_penalty=PENALTY_EXACT,
                )
                for canonical in affected:
                    if canonical != best_canonical:
                        auto_merge.append(
                            (canonical, best_canonical, match.auto_trigger())
                        )
                        cap_merges.append((canonical, best_canonical))
        else:
            # Not safe capitalization variants - add warning with full context
            contested_warnings.append((norm, canonical_to_variants))

    # Log Phase 1 results
    if logger and cap_merges:
        logger.info(
            f"  Phase 1 (contested variants): {len(cap_merges)} safe capitalization auto-merges "
            f"(e.g., '{cap_merges[0][0]}' → '{cap_merges[0][1]}')",
            extra={
                "merges": [{"source": s, "target": t} for s, t in cap_merges],
                "phase": "contested_variants",
                "merge_type": "capitalization",
            },
        )

    # Phase 2: Auto-merge via uncontested variants
    # Only process variants that map to a single canonical (not contested)
    norm_to_canonicals: dict[str, set[str]] = {}
    for canonical, variants in entities.items():
        for sv in variants:
            norm = normalize_for_comparison(sv.form)
            if norm in variant_map:  # Only uncontested
                norm_to_canonicals.setdefault(norm, set()).add(canonical)

    # Process each potential collision
    for norm_form, canonicals in norm_to_canonicals.items():
        if len(canonicals) <= 1:
            continue

        mapping = variant_map[norm_form]
        total_spec = mapping.variant.speculation  # Exact match penalty = 0

        # Additional validation for low-speculation matches
        if mapping.variant.speculation <= SPEC_BEFORE_PAREN:
            # Should be capitalization variants
            if not are_safe_capitalization_variants(canonicals):
                # Not safe - send to agent regardless of speculation
                for canonical in canonicals:
                    if canonical != mapping.canonical:
                        agent_review.append((canonical, mapping.canonical))
                continue

        # Decide based on speculation threshold
        if total_spec <= AUTO_MERGE_THRESHOLD:
            # Auto-merge - create match for reasoning
            match = EntityMatch(
                canonical=mapping.canonical,
                matched_variant=mapping.variant,
                query_variant=mapping.variant,  # Same for consolidation
                match_penalty=PENALTY_EXACT,
            )

            for canonical in canonicals:
                if canonical != mapping.canonical:
                    auto_merge.append(
                        (canonical, mapping.canonical, match.auto_trigger())
                    )
                    variant_merges.append((canonical, mapping.canonical))
        else:
            # Above threshold - agent review
            for canonical in canonicals:
                if canonical != mapping.canonical:
                    agent_review.append((canonical, mapping.canonical))

    # Log Phase 2 results
    if logger and variant_merges:
        logger.info(
            f"  Phase 2 (uncontested variants): {len(variant_merges)} auto-merges "
            f"(e.g., '{variant_merges[0][0]}' → '{variant_merges[0][1]}')",
            extra={
                "merges": [{"source": s, "target": t} for s, t in variant_merges],
                "phase": "uncontested_variants",
                "merge_type": "variant",
            },
        )

    # Phase 3: Auto-merge fuzzy spelling variants (tumor/tumour)
    if PENALTY_FUZZY <= AUTO_MERGE_THRESHOLD:
        canonical_list = list(entities.keys())
        for i, canon1 in enumerate(canonical_list):
            for canon2 in canonical_list[i + 1 :]:
                # Use core matching logic with distance info for diagnostics
                matched, dist = entity_names_match(canon1, canon2)
                if matched:
                    # Prefer shorter/simpler name as canonical (general over specific)
                    canonical, alias = min(
                        (canon1, canon2),
                        (canon2, canon1),
                        key=lambda p: (len(p[0]), p[0]),
                    )
                    auto_merge.append(
                        (
                            alias,
                            canonical,
                            f"{PENALTY_FUZZY}:original:fuzzy(dist={dist})",
                        )
                    )
                    fuzzy_merges.append((alias, canonical))

    # Log Phase 3 results
    if logger and fuzzy_merges:
        logger.info(
            f"  Phase 3 (fuzzy spelling): {len(fuzzy_merges)} auto-merges "
            f"(e.g., '{fuzzy_merges[0][0]}' → '{fuzzy_merges[0][1]}')",
            extra={
                "merges": [{"source": s, "target": t} for s, t in fuzzy_merges],
                "phase": "fuzzy_spelling",
                "merge_type": "fuzzy",
            },
        )

    # Cluster remaining entities using hierarchical clustering with IDF weighting
    from interaction_finder.extraction.clustering import (
        cluster_entities,
        compute_token_specificity,
    )

    # Exclude only merge sources (children) from clustering
    # Contested variants don't prevent canonicals from clustering - they can still
    # cluster based on other non-contested tokens
    already_handled = {child for child, parent, _ in auto_merge}

    remaining_entities = {e: entities[e] for e in entities if e not in already_handled}
    # Filter mention_counts to only remaining entities
    remaining_mentions = (
        {e: mention_counts[e] for e in remaining_entities if e in mention_counts}
        if mention_counts
        else None
    )
    clusters, merge_trees = cluster_entities(
        remaining_entities,
        threshold=cluster_threshold,
        mention_counts=remaining_mentions,
    )

    # Groups of 2+ entities for agent review
    agent_review_groups = [cluster for cluster in clusters if len(cluster) > 1]

    # Collect clustering metadata for debugging
    clustering_info = None
    if remaining_entities:
        specificity = compute_token_specificity(remaining_entities, remaining_mentions)
        cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
        large_clusters = [
            sorted(list(cluster)) for cluster in clusters if len(cluster) >= 5
        ]

        # Get top 10 most common tokens by frequency
        from collections import Counter

        token_freq = Counter()
        for variants in remaining_entities.values():
            from interaction_finder.extraction.clustering import tokenize

            for var in variants:
                token_freq.update(tokenize(var.form))

        sample_weights = {
            token: specificity.get(token, 0.0)
            for token, _ in token_freq.most_common(10)
        }

        clustering_info = {
            "total_entities": len(entities),
            "entities_in_clustering": len(remaining_entities),
            "threshold": cluster_threshold,
            "token_weights": dict(specificity),  # Full weights for analysis
            "weight_range": (
                (min(specificity.values()), max(specificity.values()))
                if specificity
                else (0.0, 0.0)
            ),
            "clusters_formed": len(clusters),
            "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
            "multi_entity_clusters": len(agent_review_groups),
            "singleton_clusters": sum(1 for c in clusters if len(c) == 1),
            "cluster_sizes": cluster_sizes,
            "large_clusters": large_clusters,
            "merge_trees": [
                tree.to_dict() for tree in merge_trees if len(tree.entities) > 1
            ],
            "sample_weights": sample_weights,
        }

    # Auto-merge summary logging (after all phases complete)
    if logger:
        total_auto_merges = len(auto_merge)
        if total_auto_merges > 0:
            logger.info(
                f"  Total auto-merges: {total_auto_merges} "
                f"({len(cap_merges)} capitalization, {len(variant_merges)} variant, {len(fuzzy_merges)} fuzzy)",
                extra={
                    "total_auto_merges": total_auto_merges,
                    "by_type": {
                        "capitalization": len(cap_merges),
                        "variant": len(variant_merges),
                        "fuzzy": len(fuzzy_merges),
                    },
                },
            )

    return ConsolidationCandidates(
        auto_merge=auto_merge,
        agent_review=agent_review,
        agent_review_groups=agent_review_groups,
        contested_warnings=contested_warnings,
        merge_trees=[t for t in merge_trees if len(t.entities) > 1],
        clustering_info=clustering_info,
    )
