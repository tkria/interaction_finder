"""Helper functions for the extraction pipeline.

Provides utilities for entity validation, proximal set identification,
text region construction, pair key generation, and markdown processing.
"""

import re
from collections import Counter

# Matches ATX-style markdown headings (# Heading)
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s", re.MULTILINE)

from interaction_finder.extraction.models import (
    EntityMention,
    EntityPairKey,
    PairAssessment,
    PairSpread,
    ProximalEntitySet,
)
from interaction_finder.resources import Resource, ResourceQuote
from interaction_finder.text_mapping import NormalizedTextMapper


def build_permitted_pairs(entity_types: list[str]) -> dict[str, set[str]]:
    """Build mapping of which entity kinds can pair with which.

    The rule: a kind must appear at least twice in the input list to permit
    self-pairs (kind-kind). Cross-pairs (kindA-kindB) are permitted if both
    kinds appear at least once.

    Special case: if only one unique kind is provided, self-pairs are allowed
    regardless of count (otherwise no pairs would be possible).

    Parameters:
        entity_types: List of entity kinds (may contain duplicates)

    Returns:
        Dict mapping each kind to the set of kinds it can pair with

    Examples:
        >>> build_permitted_pairs(["gene", "disease"])
        {'gene': {'disease'}, 'disease': {'gene'}}

        >>> build_permitted_pairs(["gene", "gene", "disease"])
        {'gene': {'gene', 'disease'}, 'disease': {'gene'}}

        >>> build_permitted_pairs(["gene"])
        {'gene': {'gene'}}

        >>> build_permitted_pairs(["gene", "disease", "protein"])
        {'gene': {'disease', 'protein'}, 'disease': {'gene', 'protein'}, 'protein': {'gene', 'disease'}}
    """
    # Count occurrences of each kind
    counts = Counter(entity_types)
    unique_kinds = set(entity_types)

    # Build permitted pairs map
    permitted: dict[str, set[str]] = {}
    for kind in unique_kinds:
        # Start with all other kinds
        allowed = unique_kinds - {kind}
        # Add self if kind appears at least twice OR if it's the only kind
        if counts[kind] >= 2 or len(unique_kinds) == 1:
            allowed.add(kind)
        permitted[kind] = allowed

    return permitted


def strip_kind_annotation(entity_name: str) -> str:
    """Strip kind annotation from entity name if present.

    Removes trailing patterns like " (gene)", " (phenotype)", " (PAH)", etc. that may
    have been incorrectly included by the LLM despite instructions. Now handles both
    lowercase kind annotations and uppercase abbreviations in parentheses.

    Parameters:
        entity_name: Entity name that may contain kind annotation

    Returns:
        Entity name with kind annotation removed

    Examples:
        >>> strip_kind_annotation("BRCA1 (gene)")
        'BRCA1'
        >>> strip_kind_annotation("Iron deficiency (phenotype)")
        'Iron deficiency'
        >>> strip_kind_annotation("Pulmonary arterial hypertension (PAH)")
        'Pulmonary arterial hypertension'
        >>> strip_kind_annotation("BRCA1")
        'BRCA1'
    """
    # Match pattern: " (word/abbreviation)" at end of string
    # Handles lowercase kind names and uppercase abbreviations
    return re.sub(r"\s+\([a-zA-Z0-9_\s-]+\)\s*$", "", entity_name).strip()


def normalize_for_comparison(text: str) -> str:
    """Normalize text for entity name comparison.

    Uses the same normalization as fuzzy quote matching to ensure consistent
    comparison behavior across the pipeline.

    Parameters:
        text: Text to normalize

    Returns:
        Normalized lowercase text
    """
    return NormalizedTextMapper.normalize(text)


def osa_distance(a: str, b: str) -> int:
    """Calculate Optimal String Alignment (restricted Damerau-Levenshtein) distance.

    Returns the minimum number of edits (insertion, deletion, substitution, or
    transposition of adjacent characters) required to transform a to b, with the
    restriction that no substring is edited more than once.

    This is particularly useful for catching spelling variants where characters
    are swapped or slightly modified (e.g., "hemorrhagic" vs "haemorrhagic").

    Parameters:
        a: First string
        b: Second string

    Returns:
        Edit distance between strings

    Examples:
        >>> osa_distance("typo", "tpyo")
        1
        >>> osa_distance("hemorrhagic", "haemorrhagic")
        2
        >>> osa_distance("frog", "cat")
        4
    """
    # Ensure a is shorter for efficiency
    if len(a) > len(b):
        a, b = b, a

    # Skip common prefix
    start = 0
    for i, (char_a, char_b) in enumerate(zip(a, b)):
        if char_a == char_b:
            start += 1
        else:
            break

    # If a is prefix of b, distance is remaining length of b
    if start == len(a):
        return len(b) - start

    # Initialize distance vectors
    len_b_tail = len(b) - start
    v0 = list(range(1, len_b_tail + 1))
    v1 = [0] * len_b_tail

    a_prev = a[0] if a else ""
    b_prev = b[0] if b else ""
    current = 0

    for i, a_char in enumerate(a):
        if i < start:
            a_prev = a_char
            continue

        left = i - start
        current = i - start + 1
        transition_next = 0

        for j, b_char in enumerate(b):
            if j < start:
                b_prev = b_char
                continue

            above = current
            this_transition = transition_next
            transition_next = v1[j - start]
            v1[j - start] = current = left
            left = v0[j - start]

            if a_char != b_char:
                # Minimum of: substitution, deletion, insertion
                current = min(current + 1, above + 1, left + 1)
                # Check for transposition
                if i > start and j > start and a_char == b_prev and a_prev == b_char:
                    current = min(current, this_transition + 1)

            v0[j - start] = current
            b_prev = b_char

        a_prev = a_char

    return current


def _expand_slash(text: str) -> list[str]:
    """Expand slash patterns in biological names.

    Handles suffix patterns (GDF1/2 → GDF1, GDF2) and simple alternation
    (TGF-β/BMP → TGF-β, BMP). Returns original first, then expansions.

    Suffix pattern detected when right side is short (≤3 chars) or Roman numerals,
    and left side ends with digits or Roman numerals that form the prefix.
    """
    if text.count("/") != 1:
        return [text]
    left, right = (s.strip() for s in text.split("/"))
    if not left or not right:
        return [text]
    # Check for suffix pattern: short right side or Roman numerals
    is_suffix = len(right) <= 3 or re.fullmatch(r"[IVX]+", right)
    if is_suffix:
        # Extract prefix from left: "GDF1" → ("GDF", "1")
        match = re.match(r"^(.+?)(\d+|[IVX]+)$", left)
        if match:
            prefix = match.group(1)
            return [text, left, prefix + right]
    # Simple alternation
    return [text, left, right]


def _extract_query_variants(query: str) -> list[str]:
    """Extract matching variants from query string in priority order.

    Handles parentheticals (base (content) → base, content) and slash patterns.
    Returns deduplicated list: original first, then base forms, then parenthetical content.
    """
    result: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        text = text.strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)

    query = query.strip()
    # Always include original query first (handles exact match with parentheses)
    add(query)
    # Split on parenthetical if present
    paren_match = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", query)
    if paren_match:
        parts = [paren_match.group(1).strip(), paren_match.group(2).strip()]
    else:
        parts = [query]
    # Add each part with slash expansion
    for part in parts:
        for expanded in _expand_slash(part):
            add(expanded)
    return result


def find_best_entity_match(
    query: str,
    candidates: list[str],
) -> str | None:
    """Find best matching candidate for a query using progressive fallbacks.

    1. Exact match against query variants
    2. Normalized exact match (case/punctuation insensitive)
    3. Fuzzy match with OSA distance, requiring specificity (gap to second-best)

    Query variants are extracted from parentheticals and slash patterns:
    - "BRCA1 (gene)" → ["BRCA1 (gene)", "BRCA1", "gene"]
    - "GDF1/2" → ["GDF1/2", "GDF1", "GDF2"]
    - "PAH (Pulmonary arterial hypertension)" → ["PAH (Pulmonary arterial hypertension)", "PAH", "Pulmonary arterial hypertension"]

    Fuzzy matching uses similarity scores (1 - distance/length) so that high-quality
    matches on long strings beat low-quality matches on short strings. Max distance
    scales with length (1 + floor(len/10)). Specificity gap scales inversely with
    match quality: gap_needed = 2 * (1 - best_similarity).
    """
    if not candidates:
        return None
    query_variants = _extract_query_variants(query)
    if not query_variants:
        return None
    # Build lookup: normalized form → original candidate
    # Two passes: variants first, then originals, so originals always win
    norm_to_original: dict[str, str] = {}
    for c in candidates:
        for c_variant in _extract_query_variants(c)[1:]:  # Skip original (index 0)
            c_norm = normalize_for_comparison(c_variant)
            if c_norm not in norm_to_original:
                norm_to_original[c_norm] = c
    for c in candidates:
        norm_to_original[normalize_for_comparison(c)] = c  # Originals overwrite
    # Normalize query variants once
    variant_norms = [normalize_for_comparison(v) for v in query_variants]
    # Stage 1+2: Exact/normalized match (query variants checked in priority order)
    for v_norm in variant_norms:
        if v_norm in norm_to_original:
            return norm_to_original[v_norm]
    # Stage 3: Fuzzy match using similarity scores
    # Similarity = 1 - distance/length, so longer matches with small distances score higher
    # This ensures a 1-edit match on 30 chars (~97% similar) beats a 3-edit match on 4 chars (~25%)
    scores: list[
        tuple[float, int, int, str]
    ] = []  # (similarity, dist, shorter_len, original)
    for c_norm, c_orig in norm_to_original.items():
        best_similarity = float("-inf")
        best_dist = 0
        best_shorter_len = 0
        for v_norm in variant_norms:
            dist = osa_distance(v_norm, c_norm)
            shorter_len = min(len(v_norm), len(c_norm))
            similarity = 1 - dist / shorter_len if shorter_len > 0 else 0
            if similarity > best_similarity:
                best_similarity = similarity
                best_dist = dist
                best_shorter_len = shorter_len
        scores.append((best_similarity, best_dist, best_shorter_len, c_orig))
    scores.sort(reverse=True)  # Higher similarity first
    best_similarity, best_dist, shorter_len, best_candidate = scores[0]
    # Require reasonable similarity (at least 70% match)
    if best_similarity < 0.7:
        return None
    # Max distance scales with string length: 1 + floor(len/10)
    max_distance = 1 + shorter_len // 10
    if best_dist > max_distance:
        return None
    # Reject matches where only short (1-2 digit) numbers differ (e.g. SMAD1/SMAD2)
    if _only_short_number_difference(query, best_candidate):
        return None
    # Require specificity: gap to second-best scales inversely with match quality
    # Gap needed = 2 * (1 - best_similarity), so strong matches need small gaps
    min_gap = 2 * (1 - best_similarity)
    if len(scores) > 1 and best_similarity - scores[1][0] < min_gap:
        return None
    return best_candidate


# Regex for 1-2 digit numbers not adjacent to other digits
_SHORT_NUMBER_RE = re.compile(r"(?<!\d)\d{1,2}(?!\d)")


def _only_short_number_difference(a: str, b: str) -> bool:
    """Check if strings differ only in short (1-2 digit) numbers.

    Used to reject fuzzy matches between entities like SMAD1/SMAD2, IL-6/IL-8,
    p53/p63 which are distinct entities differing only by number, not typos.
    """
    a_lower, b_lower = a.lower(), b.lower()
    if a_lower == b_lower:
        return False
    a_masked = _SHORT_NUMBER_RE.sub("#", a_lower)
    b_masked = _SHORT_NUMBER_RE.sub("#", b_lower)
    return a_masked == b_masked and a_masked != a_lower


def _is_valid_entity_form(form: str) -> bool:
    """Check if a form is valid as a standalone entity name.

    Filters out:
    - Pure numbers (including with spaces/dashes)
    - Very short forms (< 3 characters total)
    - Forms with list separators (commas, semicolons, multiple slashes)
    - Forms without sufficient alphabetic content or starting with numbers
    - Forms with parenthetical content (should be expanded separately)

    Parameters:
        form: Entity form to validate

    Returns:
        True if form is valid as an entity name

    Examples:
        >>> _is_valid_entity_form("BRCA1")
        True
        >>> _is_valid_entity_form("1")
        False
        >>> _is_valid_entity_form("1, 5, 8")
        False
        >>> _is_valid_entity_form("X")
        False
        >>> _is_valid_entity_form("PAH")
        True
        >>> _is_valid_entity_form("p53")
        True
        >>> _is_valid_entity_form("1a2")
        False
        >>> _is_valid_entity_form("BRCA1 (gene)")
        False
    """
    form = form.strip()
    # Reject empty strings
    if not form:
        return False
    # Reject forms with parenthetical content (these should be expanded, not kept whole)
    # This catches both kind annotations like "(gene)" and should-be-expanded forms
    if "(" in form and ")" in form:
        return False
    # Reject pure numbers (including with spaces/dashes)
    if re.match(r"^[\d\s\-]+$", form):
        return False
    # Reject forms starting with a digit (like "1a2", "123abc")
    # Valid gene names like "p53" start with a letter
    if form[0].isdigit():
        return False
    # Reject forms with list separators (commas, semicolons, multiple slashes)
    if "," in form or ";" in form or form.count("/") > 1:
        return False
    # Count alphabetic characters and check length requirements
    alpha_count = sum(1 for c in form if c.isalpha())
    # For short forms (3-4 chars), require at least 1 alphabetic character
    # For longer forms, require at least 2 alphabetic characters
    # This allows "p53" but combined with the digit-start check, rejects "1a2"
    if len(form) <= 4:
        min_alpha = 1
    else:
        min_alpha = 2
    if alpha_count < min_alpha:
        return False
    # Reject very short forms (< 3 characters)
    if len(form) < 3:
        return False
    return True


def extract_all_forms(entity_name: str, aliases: list[str]) -> list[str]:
    """Extract all distinct forms an entity can take.

    Expands entity name and aliases by extracting content from parenthetical
    forms like "Name (abbreviation)". Filters out likely kind annotations
    (single lowercase words in parens), list-like content, and invalid forms.

    Parameters:
        entity_name: Canonical entity name
        aliases: List of alias forms

    Returns:
        List of all distinct forms (name, aliases, expanded parentheticals)

    Examples:
        >>> extract_all_forms("PAH (Pulmonary arterial hypertension)", [])
        ['PAH', 'PAH (Pulmonary arterial hypertension)', 'Pulmonary arterial hypertension']

        >>> extract_all_forms("Telangiectasia", ["HHT"])
        ['HHT', 'Telangiectasia']

        >>> extract_all_forms("BRCA1 (gene)", [])
        ['BRCA1']

        >>> extract_all_forms("R-SMADs (1, 5, 8)", [])
        ['R-SMADs']

        >>> extract_all_forms("ACVRL1", [])
        ['ACVRL1']
    """
    all_forms = {entity_name}
    all_forms.update(aliases)

    # Expand parenthetical forms
    expanded = set()
    for form in all_forms:
        # Always add the original form if valid
        if _is_valid_entity_form(form):
            expanded.add(form)

        # Check for parenthetical content
        match = re.match(r"^(.+?)\s*\(([^)]+)\)$", form.strip())
        if match:
            base = match.group(1).strip()
            paren_content = match.group(2).strip()

            # Add the base without parens if valid
            if _is_valid_entity_form(base):
                expanded.add(base)

            # Add paren content if it looks like an abbreviation/alternative name
            # Skip:
            # - Single lowercase words (likely kind annotations like "(gene)", "(phenotype)")
            # - Invalid forms (numbers, lists, etc.)
            if (
                paren_content
                and not (paren_content.islower() and " " not in paren_content)
                and _is_valid_entity_form(paren_content)
            ):
                expanded.add(paren_content)

    return sorted(expanded)


def is_obvious_variant(a: str, b: str) -> bool:
    """Check if strings are obvious variants (plurals or common spelling differences).

    Detects:
    - Plural patterns: +s, +es, y→ies
    - US/UK spelling: ae↔e, our↔or, ise↔ize, re↔er

    Parameters:
        a: First string (normalized)
        b: Second string (normalized)

    Returns:
        True if they match known variant patterns

    Examples:
        >>> is_obvious_variant("telangiectasia", "telangiectasias")
        True
        >>> is_obvious_variant("haemorrhagic", "hemorrhagic")
        True
        >>> is_obvious_variant("colour", "color")
        True
        >>> is_obvious_variant("cat", "dog")
        False
    """
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)

    # Plural patterns
    if longer == shorter + "s" or longer == shorter + "es":
        return True
    if shorter.endswith("y") and longer == shorter[:-1] + "ies":
        return True

    # US/UK spelling variants (bidirectional)
    for pattern_from, pattern_to in [("ae", "e"), ("our", "or")]:
        if pattern_from in a and b == a.replace(pattern_from, pattern_to):
            return True
        if pattern_from in b and a == b.replace(pattern_from, pattern_to):
            return True

    # Suffix variants
    if a.endswith("ise") and b == a[:-3] + "ize":
        return True
    if b.endswith("ise") and a == b[:-3] + "ize":
        return True
    if a.endswith("re") and len(a) > 3 and b == a[:-2] + "er":
        return True
    if b.endswith("re") and len(b) > 3 and a == b[:-2] + "er":
        return True

    return False


def find_substring_entities(
    entities: dict[str, EntityMention],
) -> list[tuple[str, str]]:
    """Find entity pairs where normalized names are equal or one is a substring of another.

    Compares normalized lowercase versions of entity names to identify
    potential merge candidates (e.g., "BRCA" and "BRCA1", or "PAH" and "pah").

    Parameters:
        entities: Dict mapping canonical name to EntityMention

    Returns:
        List of (parent_name, child_name) tuples where:
        - For exact normalized matches: parent is the original (keeps first seen)
        - For substring matches: parent is the shorter/more general name, child is longer/more specific
    """
    candidates = []
    entity_names = list(entities.keys())

    # Compare each pair of entities
    for i, name1 in enumerate(entity_names):
        norm1 = normalize_for_comparison(name1)

        for name2 in entity_names[i + 1 :]:
            norm2 = normalize_for_comparison(name2)

            # Check if normalized forms are identical
            if norm1 == norm2:
                # Exact match after normalization → keep first, merge second
                candidates.append((name1, name2))
            # Check if either is a substring of the other (but not equal)
            elif norm1 in norm2:
                # name1 is substring of name2 → name1 is parent (general), name2 is child (specific)
                candidates.append((name1, name2))
            elif norm2 in norm1:
                # name2 is substring of name1 → name2 is parent (general), name1 is child (specific)
                candidates.append((name2, name1))

    return candidates


def identify_proximal_sets(
    entities: dict[str, EntityMention], threshold: int, resource: Resource
) -> list[ProximalEntitySet]:
    """Identify groups of entities found in close proximity using sliding window.

    Algorithm:
    1. Get chunk indices for each entity's quotes
    2. Sort quotes by (start_chunk, -length_in_chunks)
    3. Use sliding window with configurable threshold
    4. Group entities whose quotes appear within threshold chunks of each other

    Parameters:
        entities: Dict mapping canonical name to EntityMention
        threshold: Maximum chunk distance to consider entities proximal
        resource: Resource containing chunk boundary information

    Returns:
        List of ProximalEntitySet objects, each containing entities found near each other
    """
    if not entities:
        return []

    # Build list of (entity_name, quote, start_chunk, end_chunk)
    quote_info = []
    for entity_name, entity in entities.items():
        for quote in entity.quotes:
            # Get chunk indices for this quote
            chunk_indices = quote.chunk_indices
            if not chunk_indices:
                # Quote doesn't overlap any chunks, skip it
                continue

            start_chunk = min(chunk_indices)
            end_chunk = max(chunk_indices)
            quote_info.append((entity_name, quote, start_chunk, end_chunk))

    if not quote_info:
        return []

    # Sort by (start_chunk, -length) to process quotes in order
    quote_info.sort(key=lambda x: (x[2], -(x[3] - x[2])))

    # Sliding window algorithm
    proximal_sets = []
    current_entities: set[str] = set()
    current_quotes: dict[str, list[ResourceQuote]] = {}
    window_start = quote_info[0][2]
    window_end = quote_info[0][3]

    for entity_name, quote, start_chunk, end_chunk in quote_info:
        # Check if quote starts within current window + threshold
        if start_chunk <= window_end + threshold:
            # Add to current proximal set
            current_entities.add(entity_name)
            if entity_name not in current_quotes:
                current_quotes[entity_name] = []
            current_quotes[entity_name].append(quote)

            # Expand window to include this quote
            window_end = max(window_end, end_chunk)
        else:
            # Quote is too far away, finalize current set
            if len(current_entities) >= 2:  # Only keep sets with 2+ entities
                proximal_sets.append(
                    ProximalEntitySet(
                        entities=current_entities.copy(),
                        chunk_range=(window_start, window_end),
                        entity_quotes=current_quotes.copy(),
                    )
                )

            # Start new proximal set
            current_entities = {entity_name}
            current_quotes = {entity_name: [quote]}
            window_start = start_chunk
            window_end = end_chunk

    # Don't forget the last set
    if len(current_entities) >= 2:
        proximal_sets.append(
            ProximalEntitySet(
                entities=current_entities.copy(),
                chunk_range=(window_start, window_end),
                entity_quotes=current_quotes.copy(),
            )
        )

    return proximal_sets


def build_text_region(
    resource: Resource, chunk_start: int, chunk_end: int, padding: int
) -> str:
    """Build text region from chunk range with padding.

    Extracts all chunks from (chunk_start - padding) to (chunk_end + padding),
    concatenating them into a single text string.

    Parameters:
        resource: Resource containing the text and chunks
        chunk_start: Starting chunk index (inclusive)
        chunk_end: Ending chunk index (inclusive)
        padding: Number of chunks to add on each side

    Returns:
        Concatenated text from selected chunks
    """
    # Apply padding with bounds checking
    start_idx = max(0, chunk_start - padding)
    end_idx = min(len(resource.chunks) - 1, chunk_end + padding)

    # Extract chunks
    chunks = []
    for chunk_idx in range(start_idx, end_idx + 1):
        chunk_start_pos, chunk_end_pos = resource.chunks[chunk_idx]
        chunk_text = resource.text[chunk_start_pos:chunk_end_pos]
        chunks.append(chunk_text)

    # Concatenate chunks
    return "\n".join(chunks)


def collect_relevant_text_for_quotes(
    resource: Resource, quotes: list[ResourceQuote], padding: int
) -> str:
    """Build text region containing all provided quotes with padding.

    Identifies the chunk range spanning all quotes and returns the contiguous
    text from start to end (including padding).

    Parameters:
        resource: Resource containing the text and chunks
        quotes: List of quotes to cover
        padding: Number of chunks to add on each side

    Returns:
        Contiguous text covering all quotes with padding
    """
    if not quotes:
        return ""

    # Collect all chunk indices from all quotes
    all_chunks = set()
    for quote in quotes:
        all_chunks.update(quote.chunk_indices)

    if not all_chunks:
        return ""

    # Get range from min to max chunk with padding
    start_idx = max(0, min(all_chunks) - padding)
    end_idx = min(len(resource.chunks) - 1, max(all_chunks) + padding)

    # Extract all chunks in the range
    text_parts = []
    for chunk_idx in range(start_idx, end_idx + 1):
        chunk_start_pos, chunk_end_pos = resource.chunks[chunk_idx]
        chunk_text = resource.text[chunk_start_pos:chunk_end_pos]
        text_parts.append(chunk_text)

    return "\n".join(text_parts)


def make_entity_pair_key(
    entity1: EntityMention, entity2: EntityMention
) -> EntityPairKey:
    """Create consistent EntityPairKey for two entities.

    Orders entities lexicographically by their kinds to ensure consistent
    pairing (e.g., always gene-disease, not disease-gene).

    Parameters:
        entity1: First entity
        entity2: Second entity

    Returns:
        EntityPairKey with entities ordered by kind
    """
    # Order by kind lexicographically
    if entity1.kind < entity2.kind:
        return EntityPairKey(entity1_name=entity1.name, entity2_name=entity2.name)
    elif entity1.kind > entity2.kind:
        return EntityPairKey(entity1_name=entity2.name, entity2_name=entity1.name)
    else:
        # Same kind - order by name lexicographically
        if entity1.name < entity2.name:
            return EntityPairKey(entity1_name=entity1.name, entity2_name=entity2.name)
        else:
            return EntityPairKey(entity1_name=entity2.name, entity2_name=entity1.name)


# =============================================================================
# Relationship polarity utilities
# =============================================================================


def get_relationship_polarity(relationship: str, polarity_map: dict[str, str]) -> str:
    """Look up polarity for a relationship label.

    Parameters:
        relationship: Relationship label to look up
        polarity_map: Mapping from relationship to polarity

    Returns:
        Polarity category: "supporting", "refuting", "neutral", or "irrelevant"

    Raises:
        KeyError: If relationship not in mapping (indicates consolidation bug)
    """
    return polarity_map[relationship]


def build_pair_spread(
    assessments: list[PairAssessment], polarity_map: dict[str, str]
) -> PairSpread:
    """Group assessments by relationship polarity.

    Creates a PairSpread by looking up the polarity of each assessment's
    relationship label and organizing them into supporting/refuting/neutral/irrelevant
    categories.

    Parameters:
        assessments: All per-document assessments for one entity pair
        polarity_map: Mapping from relationship label to polarity

    Returns:
        PairSpread with assessments organized by polarity

    Raises:
        KeyError: If any assessment's relationship is not in polarity_map
    """
    spread = PairSpread()

    for assessment in assessments:
        polarity = get_relationship_polarity(assessment.relationship, polarity_map)
        getattr(spread, polarity).append(assessment)

    return spread


# =============================================================================
# Markdown utilities
# =============================================================================


def adjust_heading_levels(text: str, target_min_level: int) -> str:
    """Adjust all markdown heading levels so the minimum equals target_min_level.

    Finds the minimum heading level in the text (e.g., level 1 for '# Heading')
    and shifts all headings up or down so that minimum becomes target_min_level.
    Headings that would exceed level 6 are clamped to level 6.

    Parameters:
        text: Markdown text potentially containing headings
        target_min_level: Desired minimum heading level (1-6)

    Returns:
        Text with adjusted heading levels

    Examples:
        >>> adjust_heading_levels("# Title\\n## Section", target_min_level=2)
        '## Title\\n### Section'

        >>> adjust_heading_levels("## Already level 2", target_min_level=2)
        '## Already level 2'

        >>> adjust_heading_levels("### Deep heading", target_min_level=1)
        '# Deep heading'
    """
    if not 1 <= target_min_level <= 6:
        raise ValueError(f"target_min_level must be 1-6, got {target_min_level}")
    # Find all heading levels in the text
    matches = _HEADING_PATTERN.findall(text)
    if not matches:
        return text
    # Determine current minimum level and required shift
    current_min = min(len(hashes) for hashes in matches)
    shift = target_min_level - current_min
    if shift == 0:
        return text

    # Replace each heading with adjusted level
    def adjust_heading(match: re.Match) -> str:
        hashes = match.group(1)
        new_level = min(6, max(1, len(hashes) + shift))
        return "#" * new_level + " "

    return _HEADING_PATTERN.sub(adjust_heading, text)


# Pattern for document ID citations: [N_hash] where N is a number and hash is exactly 8 lowercase alphanumeric chars
CITATION_PATTERN = re.compile(r"\[(\d+)_([a-z0-9]{8})\]")


def extract_document_citations(text: str) -> list[str]:
    """Extract document ID citations from text.

    Finds all citations in the format [N_hash] where N is a counter
    and hash is the alphanumeric resource hash.

    Parameters:
        text: Text potentially containing citations

    Returns:
        List of unique document IDs in order of first appearance

    Examples:
        >>> extract_document_citations("Evidence from [1_abc12345] and [2_def67890]")
        ['1_abc12345', '2_def67890']

        >>> extract_document_citations("Cited [1_abc12345] twice [1_abc12345]")
        ['1_abc12345']
    """
    seen = set()
    result = []
    for match in CITATION_PATTERN.finditer(text):
        doc_id = f"{match.group(1)}_{match.group(2)}"
        if doc_id not in seen:
            seen.add(doc_id)
            result.append(doc_id)
    return result


def validate_document_citations(
    cited_ids: list[str], valid_ids: set[str]
) -> tuple[list[str], list[str]]:
    """Validate cited document IDs against a set of valid IDs.

    Parameters:
        cited_ids: Document IDs extracted from text
        valid_ids: Set of document IDs that were provided in the prompt

    Returns:
        (valid_citations, invalid_citations) tuple
    """
    valid = [doc_id for doc_id in cited_ids if doc_id in valid_ids]
    invalid = [doc_id for doc_id in cited_ids if doc_id not in valid_ids]
    return valid, invalid
