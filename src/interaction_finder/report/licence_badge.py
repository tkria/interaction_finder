"""Licence and redaction notices for a document in a report.

A report shows two things about how a source document reached the reader: the
licence the article carries, and --- where its text could not be redistributed
--- the fact that what follows is a quote skeleton rather than the article.

Both belong with the document's provenance links, beside "View original" and the
DOI, since that is where a reader looks to establish what they are reading.

Public API:
    licence_url         licence name -> canonical deed URL, if one exists
    licence_badge_html  licence name -> badge markup (a link where possible)
    redaction_notice_html  resource fields -> notice shown above a skeleton
"""

from __future__ import annotations

from html import escape

__all__ = [
    "licence_badge_html",
    "licence_url",
    "redaction_notice_html",
]

# Canonical deeds for the licence names OpenAlex reports. Version 4.0 is the
# current CC release; a work under an earlier version is still described by its
# deed, and linking the current one is the convention publishers follow.
_LICENCE_URLS = {
    "cc-by": "https://creativecommons.org/licenses/by/4.0/",
    "cc-by-sa": "https://creativecommons.org/licenses/by-sa/4.0/",
    "cc-by-nc": "https://creativecommons.org/licenses/by-nc/4.0/",
    "cc-by-nd": "https://creativecommons.org/licenses/by-nd/4.0/",
    "cc-by-nc-nd": "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    "cc-by-nc-sa": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    "cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "public-domain": "https://creativecommons.org/publicdomain/mark/1.0/",
}

_LICENCE_NAMES = {
    "cc-by": "CC BY",
    "cc-by-sa": "CC BY-SA",
    "cc-by-nc": "CC BY-NC",
    "cc-by-nd": "CC BY-ND",
    "cc-by-nc-nd": "CC BY-NC-ND",
    "cc-by-nc-sa": "CC BY-NC-SA",
    "cc0": "CC0",
    "public-domain": "Public domain",
    "other-oa": "Open access",
}

_LICENCE_TOOLTIP = (
    "Redistribution licence of the source article, as recorded by OpenAlex"
)


def licence_url(licence: str | None) -> str | None:
    """Canonical deed URL for a licence name, or None if it names no deed."""
    if not licence:
        return None
    return _LICENCE_URLS.get(licence.strip().lower())


def licence_badge_html(licence: str | None) -> str:
    """Badge markup for a licence, linking its deed where one exists.

    Renders as a link for Creative Commons licences and as a plain span
    otherwise (``other-oa`` names no specific licence, so there is nothing to
    link to). Returns "" when no licence is recorded.
    """
    if not licence or not licence.strip():
        return ""
    key = licence.strip().lower()
    label = escape(_LICENCE_NAMES.get(key, licence.strip()))
    url = _LICENCE_URLS.get(key)
    if url is None:
        return (
            f'<span class="document-licence" title="{_LICENCE_TOOLTIP}">{label}</span>'
        )
    return (
        f'<a class="document-licence" href="{escape(url)}" target="_blank"'
        f' rel="noreferrer noopener" title="{_LICENCE_TOOLTIP}">{label}</a>'
    )


def redaction_notice_html(
    text_length: int | None = None, licence: str | None = None
) -> str:
    """Notice explaining that a document is shown as a quote skeleton.

    text_length: characters of article text withheld, if recorded, so a reader
        can see the scale of what is not shown.
    licence: the article's licence, named in the explanation when it identifies
        one. Placeholders such as ``other-oa`` name no licence, so the notice
        says the terms could not be established rather than asserting they
        forbid redistribution.

    Returns: HTML for the notice, placed between a document's provenance links
        and its content.
    """
    named = licence.strip() if licence and licence.strip() else ""
    reason = (
        f"its licence ({escape(named)}) does not permit redistributing it"
        if named.lower() in _LICENCE_URLS
        else "no licence permitting redistribution could be established for it"
    )
    withheld = (
        f" The full text runs to about {text_length:,} characters."
        if text_length
        else ""
    )
    return (
        '<div class="document-redaction-notice" role="note">'
        '<span class="document-redaction-icon" aria-hidden="true">&#9432;</span>'
        "<span><strong>Shown as quoted extracts.</strong> The text of this "
        f"article is not reproduced here because {reason}. What follows are the "
        "passages this report quotes, in their original sections, with the "
        f"length of omitted material marked.{withheld} Use "
        "&ldquo;View original&rdquo; above to read the article at its source."
        "</span></div>"
    )
