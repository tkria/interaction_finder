from __future__ import annotations
import re
from typing import List
from .models import Term


def parse_term_line(line: str) -> Term | None:
    """
    Parse a single line into a Term object.

    Supports formats:
    - termname # comment
    - termname # &attr1=value1, &attr2=value2, ...
    - termname # &attr1=value1 &attr2=value2
    - termname # comment &attr=value
    - termname # &attr1=multi word value &attr2=val

    If &kind is present, it sets the kind field. Other &attributes go into the attributes dict.
    Commas between attributes are optional.

    Args:
        line: A line of text to parse

    Returns:
        Term object if line contains a valid term, None if line is empty or comment-only
    """
    # Strip whitespace
    line = line.strip()

    # Skip empty lines
    if not line:
        return None

    # Skip pure comment lines (starting with #)
    if line.startswith("#"):
        return None

    # Split on first # to separate term name from comment/attributes
    if "#" in line:
        term_part, comment_part = line.split("#", 1)
        term_name = term_part.strip()
        comment_part = comment_part.strip()
    else:
        term_name = line
        comment_part = ""

    # Skip if no term name
    if not term_name:
        return None

    # Parse attributes from comment part
    kind = None
    attributes = {}

    if comment_part:
        # Parse attributes using a more flexible approach
        # This handles optional commas and multi-word values
        attributes_dict = _parse_attributes(comment_part)

        for attr_name, attr_value in attributes_dict.items():
            if attr_name == "kind":
                kind = attr_value
            else:
                attributes[attr_name] = attr_value

    return Term(name=term_name, kind=kind, attributes=attributes)


def _parse_attributes(text: str) -> dict[str, str]:
    """
    Parse attributes from a text string, handling optional commas and multi-word values.

    Examples:
    - "&attr1=val, &attr2=val"
    - "&attr1=val &attr2=val"
    - "&attr1=multi word value &attr2=val"
    - "&attr1=multi word value & other thing, &attr2=val2"

    Args:
        text: Text containing attributes

    Returns:
        Dictionary of attribute name -> value pairs
    """
    attributes = {}

    # Find all positions where & appears followed by word characters and =
    # This helps us identify attribute starts
    attr_starts = []
    i = 0
    while i < len(text):
        if text[i] == "&":
            # Look ahead to see if this looks like an attribute
            j = i + 1
            # Skip the attribute name (word characters)
            while j < len(text) and (text[j].isalnum() or text[j] == "_"):
                j += 1
            # Check if we have an = sign
            if j < len(text) and text[j] == "=":
                attr_starts.append(i)
        i += 1

    # Parse each attribute
    for idx, start_pos in enumerate(attr_starts):
        # Find the end position (start of next attribute or end of string)
        if idx + 1 < len(attr_starts):
            end_pos = attr_starts[idx + 1]
        else:
            end_pos = len(text)

        # Extract the attribute substring
        attr_text = text[start_pos:end_pos]

        # Parse this individual attribute
        if "=" in attr_text:
            # Find the first = sign
            eq_pos = attr_text.find("=")
            attr_name = attr_text[1:eq_pos].strip()  # Skip the & and get name
            attr_value = attr_text[eq_pos + 1 :].strip()  # Get value after =

            # Remove trailing comma if present
            if attr_value.endswith(","):
                attr_value = attr_value[:-1].strip()

            if attr_name:
                if attr_value:
                    attributes[attr_name] = attr_value
                # Skip attributes with empty values

    return attributes


def parse_terms_from_lines(lines: List[str]) -> List[Term]:
    """
    Parse terms from a list of lines.

    Args:
        lines: List of lines to parse

    Returns:
        List of Term objects
    """
    terms = []
    for line in lines:
        term = parse_term_line(line)
        if term is not None:
            terms.append(term)

    return terms
