"""Markdown text utilities."""

import re

# Matches ATX-style markdown headings (# Heading)
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s", re.MULTILINE)


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
