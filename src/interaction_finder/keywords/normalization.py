"""
Term normalization for bridging term deduplication.

Provides a multi-stage normalization pipeline that handles:
1. Parenthetical abbreviations (PAH, IPAH, SNP)
2. Greek letters, punctuation, whitespace (via NormalizedTextMapper.normalize)
3. Singular/plural via lemmatization
4. Domain-specific suffixes (pathway, signaling, in PAH)
"""

import re
from functools import lru_cache

from interaction_finder.text_mapping import NormalizedTextMapper


# Common biomedical/scientific suffixes to strip for deduplication
COMMON_SUFFIXES = [
    " pathway",
    " signaling pathway",
    " signaling",
]


def strip_parenthetical_abbreviations(text: str) -> str:
    """
    Strip parenthetical abbreviations from term.

    Removes patterns like "(PAH)", "(IPAH)", "(SNP)", etc. from the end
    or middle of terms to normalize variations.

    Parameters:
        text: str — input term

    Returns:
        str — term with parenthetical abbreviations removed

    Example:
        >>> strip_parenthetical_abbreviations("Pulmonary arterial hypertension (PAH)")
        'Pulmonary arterial hypertension'
        >>> strip_parenthetical_abbreviations("Single nucleotide polymorphism (SNP)")
        'Single nucleotide polymorphism'
    """
    # Pattern matches: space followed by opening paren, capital letters/numbers/hyphens, closing paren
    # Examples: " (PAH)", " (IPAH)", " (TGF-β)", " (SNPs)"
    pattern = r"\s*\([A-Z][A-Za-z0-9\-βαγδ]*\)"
    return re.sub(pattern, "", text).strip()


def strip_common_suffixes(text: str) -> str:
    """
    Strip common domain-specific suffixes for deduplication.

    Removes trailing phrases like " pathway", " signaling pathway" that
    create unnecessary duplicates.

    Parameters:
        text: str — normalized term

    Returns:
        str — term with common suffixes removed

    Example:
        >>> strip_common_suffixes("bmp signaling pathway")
        'bmp'
        >>> strip_common_suffixes("tgf beta signaling")
        'tgf beta'
    """
    text_lower = text.lower()
    # Try longest suffixes first to avoid partial matches
    for suffix in sorted(COMMON_SUFFIXES, key=len, reverse=True):
        if text_lower.endswith(suffix):
            return text[: -len(suffix)].strip()
    return text


@lru_cache(maxsize=1024)
def lemmatize_term(text: str) -> str:
    """
    Lemmatize term to handle singular/plural variations.

    Uses spaCy for proper linguistic lemmatization. Caches results
    since same terms appear frequently.

    Parameters:
        text: str — normalized text

    Returns:
        str — lemmatized text

    Example:
        >>> lemmatize_term("genetic risk factors")
        'genetic risk factor'
        >>> lemmatize_term("cardiovascular diseases")
        'cardiovascular disease'
    """
    try:
        import spacy

        # Try to load model, fall back to simple plural stripping if unavailable
        try:
            nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            # Model not installed, fall back to simple heuristic
            return _simple_depluralize(text)

        doc = nlp(text)
        lemmatized = " ".join(token.lemma_ for token in doc)
        return lemmatized

    except ImportError:
        # spaCy not installed, fall back to simple heuristic
        return _simple_depluralize(text)


def _simple_depluralize(text: str) -> str:
    """
    Simple plural removal as fallback when spaCy unavailable.

    Handles common English plural patterns:
    - words ending in 'ies' → 'y' (studies → study)
    - words ending in 'ses' → 's' (analyses → analysis)
    - words ending in 'ses' where stem ends in 'se' → keep 'se' (diseases → disease)
    - words ending in 's' → '' (factors → factor)

    Parameters:
        text: str — text to depluralize

    Returns:
        str — text with simple plural removal applied
    """
    words = text.split()
    result = []

    for word in words:
        # Skip very short words
        if len(word) <= 3:
            result.append(word)
            continue

        # Pattern: -ies → -y (studies → study)
        if word.endswith("ies") and len(word) > 4:
            result.append(word[:-3] + "y")
        # Pattern: -ses where stem ends in 'se' → keep 'se' (diseases → disease)
        elif word.endswith("ses") and len(word) > 5:
            stem = word[:-2]
            if stem.endswith("se"):
                result.append(stem)
            else:
                result.append(word[:-1])  # Remove just 's'
        # Pattern: -xes, -zes, -ches, -shes → remove 'es' (boxes → box)
        elif word.endswith(("xes", "zes", "ches", "shes")) and len(word) > 4:
            result.append(word[:-2])
        # Pattern: -es → try removing just 's' (genes → gene)
        elif word.endswith("es") and len(word) > 4:
            result.append(word[:-1])
        # Pattern: -s → '' (factors → factor)
        elif word.endswith("s") and not word.endswith("ss"):
            result.append(word[:-1])
        else:
            result.append(word)

    return " ".join(result)


@lru_cache(maxsize=2048)
def normalize_term_for_deduplication(term: str) -> str:
    """
    Normalize bridging term for deduplication matching.

    Applies multi-stage pipeline:
    1. Strip parenthetical abbreviations (PAH, IPAH, etc.)
    2. Text normalization (case, punctuation, Greek letters via NormalizedTextMapper)
    3. Lemmatization (singular/plural)
    4. Strip common suffixes (pathway, signaling, etc.)

    Parameters:
        term: str — raw bridging term

    Returns:
        str — normalized deduplication key

    Example:
        >>> normalize_term_for_deduplication("Pulmonary arterial hypertension (PAH)")
        'pulmonary arterial hypertension'
        >>> normalize_term_for_deduplication("Genetic risk factors")
        'genetic risk factor'
        >>> normalize_term_for_deduplication("BMP signaling pathway")
        'bmp'
        >>> normalize_term_for_deduplication("TGF-β/BMP signaling")
        'tgf beta bmp'
    """
    # Stage 1: Strip parenthetical abbreviations
    text = strip_parenthetical_abbreviations(term)
    # Stage 2: Apply text normalization (case, punctuation, Greek, whitespace)
    text = NormalizedTextMapper.normalize(text)
    # Stage 3: Lemmatize to handle plurals
    text = lemmatize_term(text)
    # Stage 4: Strip common domain suffixes
    text = strip_common_suffixes(text)
    return text.strip()
