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

    def auto_reason(self) -> str:
        """Format: auto:<total>:<source>:<match_kind>"""
        source = self.matched_variant.source
        if self.matched_variant.is_from_alias:
            source = f"alias:{source}"
        match_kind = "exact" if self.match_penalty == PENALTY_EXACT else "fuzzy"
        return f"auto:{self.total_speculation}:{source}:{match_kind}"


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
    contested_warnings: list[
        tuple[str, dict[str, list[str]]]
    ]  # (normalized_form, {canonical: [variant_forms]})


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
            if not entity_names_match(n1, n2):
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

    # Extract query variants
    query_variants = extract_entity_variants(query)

    # Stage 1: Try exact matches
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
            # Skip if too short for fuzzy matching
            shorter_len = min(len(qv_norm), len(variant_norm))
            if shorter_len < MIN_LENGTH_FOR_FUZZY:
                continue

            dist = osa_distance(qv_norm, variant_norm)
            similarity = 1 - dist / shorter_len

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

    Returns:
        True if token overlap detected
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

    # Check if tokens from one match variants of other, or if they share tokens
    return bool(tokens1 & norms2 or tokens2 & norms1 or tokens1 & tokens2)


def _select_best_canonical(canonical_names: set[str]) -> str:
    """Select best capitalization: prefer mixed-case over all-upper/all-lower."""
    return max(
        canonical_names,
        key=lambda v: (
            sum(1 for c in v if c.islower())
            * sum(1 for c in v if c.isupper()),  # Mixed
            sum(1 for c in v if c.isupper()),  # Uppercase count
            -sum(1 for c in v if c.islower()),  # Minimize lowercase
        ),
    )


def find_consolidation_candidates(
    entities: dict[str, list[SpeculatedVariant]],
) -> ConsolidationCandidates:
    """Find entity pairs for consolidation (auto-merge if ≤ threshold, else agent review)."""
    # Build variant map (excludes contested)
    variant_map, contested = build_variant_map(entities)

    # Track results
    auto_merge: list[tuple[str, str, str]] = []
    agent_review: list[tuple[str, str]] = []
    contested_warnings: list[tuple[str, dict[str, list[str]]]] = []

    # Handle contested variants: check if they're safe capitalization variants
    # If safe, pick best canonical and auto-merge. If not, add to warnings.
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
            best_canonical = _select_best_canonical(affected)

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
                            (canonical, best_canonical, match.auto_reason())
                        )
        else:
            # Not safe capitalization variants - add warning with full context
            contested_warnings.append((norm, canonical_to_variants))

    # Find potential merges by inverting variant map
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
                        (canonical, mapping.canonical, match.auto_reason())
                    )
        else:
            # Above threshold - agent review
            for canonical in canonicals:
                if canonical != mapping.canonical:
                    agent_review.append((canonical, mapping.canonical))

    # Check for fuzzy spelling variants (tumor/tumour) between remaining entities
    if PENALTY_FUZZY <= AUTO_MERGE_THRESHOLD:
        canonical_list = list(entities.keys())
        for i, canon1 in enumerate(canonical_list):
            for canon2 in canonical_list[i + 1 :]:
                # Use core matching logic
                if entity_names_match(canon1, canon2):
                    # Prefer shorter/simpler name as canonical (general over specific)
                    canonical, alias = min(
                        (canon1, canon2),
                        (canon2, canon1),
                        key=lambda p: (len(p[0]), p[0]),
                    )
                    auto_merge.append(
                        (alias, canonical, f"auto:{PENALTY_FUZZY}:original:fuzzy")
                    )

    # Bag-of-words detection pass
    canonical_list = list(entities.keys())
    already_handled: set[tuple[str, str]] = set()

    # Track already-handled pairs
    for child, parent, _ in auto_merge:
        already_handled.add((child, parent))
        already_handled.add((parent, child))
    for child, parent in agent_review:
        already_handled.add((child, parent))
        already_handled.add((parent, child))
    # Also track contested entity pairs to avoid re-proposing them
    for norm_form, canonical_to_variants in contested_warnings:
        canonicals = list(canonical_to_variants.keys())
        for i, c1 in enumerate(canonicals):
            for c2 in canonicals[i + 1 :]:
                already_handled.add((c1, c2))
                already_handled.add((c2, c1))

    # Check remaining pairs
    for i, canon1 in enumerate(canonical_list):
        for canon2 in canonical_list[i + 1 :]:
            if (canon1, canon2) in already_handled or (
                canon2,
                canon1,
            ) in already_handled:
                continue

            if _has_token_overlap(canon1, canon2, entities):
                # Order as (alias, canonical): prefer shorter/simpler name as canonical
                canonical, alias = min(
                    (canon1, canon2),
                    (canon2, canon1),
                    key=lambda p: (len(p[0]), p[0]),
                )
                agent_review.append((alias, canonical))

    return ConsolidationCandidates(
        auto_merge=auto_merge,
        agent_review=agent_review,
        contested_warnings=contested_warnings,
    )
