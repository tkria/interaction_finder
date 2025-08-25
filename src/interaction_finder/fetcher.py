"""
Enhanced PageFetcher with intelligent URL handling and granular status updates.

Features:
- Single URL and batch URL fetching with automatic concurrency
- Granular status updates showing detailed crawl4ai stages
- Automatic PDF vs HTML detection based on URL extension
- DOI extraction from HTML pages using XPath selectors
- References section removal from markdown content
- Efficient lazy imports for faster startup
- Progress bars for batch operations, detailed status for single URLs
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING, Union, List, Any
import aiofiles
import aiofiles.os
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

if TYPE_CHECKING:
    from .settings import IfetcherConfig

import re

class PreviousFailure(Exception):
    """Raised when a previous fetch failure sentinel is present for a URL."""
    def __init__(self, url: str, message: str | None = None):
        super().__init__(message or f"Previous failure recorded for URL: {url}")
        self.url = url

# Content type configurations for serialization/deserialization
CONTENT_TYPE_CONFIG = {
    "html": {"extension": "html", "serialize": str, "deserialize": str},
    "pdf": {"extension": "pdf", "serialize": str, "deserialize": str},
    "markdown": {"extension": "md", "serialize": str, "deserialize": str},
    "raw_markdown": {"extension": "raw.md", "serialize": str, "deserialize": str},
    "chunks": {
        "extension": "json",
        "serialize": lambda x: json.dumps(x, indent=2, ensure_ascii=False),
        "deserialize": lambda x: json.loads(x)
    },
    "doi": {"extension": "doi", "serialize": str, "deserialize": lambda x: x.strip()}
}

# Heading classification for academic content processing
RELEVANT_HEADINGS = {
    'abstract', 'summary', 'introduction', 'methods', 'methodology', 'results',
    'discussion', 'conclusion', 'conclusions', 'background', 'objectives',
    'findings', 'analysis', 'materials', 'procedure', 'approach',
    'appendix', 'appendices', 'limitations', 'future work', 'implications',
    'main text'
}

IRRELEVANT_HEADINGS = {
    'references', 'keywords', 'bibliography', 'citations', 'authors', 'author',
    'acknowledgements', 'acknowledgments', 'share', 'sharing', 'content link',
    'funding', 'conflicts', 'conflict of interest', 'competing interests',
    'data availability', 'supplementary', 'supporting information',
    'copyright', 'license', 'permissions', 'ethics', 'rights',
    'affiliations', 'corresponding', 'license', 'licence', 'cite',
    'how to cite', 'cite this', 'cited by', 'reprints',
    'related articles', 'similar articles', 'publication history',
    'publication types', 'notes', 'comments'
}

# Pattern lists for content classification
IRRELEVANT_PATTERNS = [
    # Site chrome and navigation (exclude paywall/login which live in PAYWALL_PATTERNS)
    r'.*cookies?.*', r'.*privacy.*', r'.*terms.*', r'.*subscribe.*',
    r'.*newsletter.*', r'.*follow.*', r'.*social.*', r'.*menu.*',
    r'.*navigation.*', r'.*search.*', r'.*contact.*', r'.*about.*',
    r'.*read.*content.*', r'similar content.*', r'.*viewed by others', r'recommended.*',
    r'supplementary\s+.*', r'supplemental\s+.*', r'.*metrics.*', r'.*altmetric.*',
    r'declaration\s+.*', r'.*availability.*', r'.*privacy.*',
    r'.*password.*', r'.*username.*', r'comment\s?.*'
]

PAYWALL_PATTERNS = [
    r'.*log ?in.*', r'.*sign\s*in.*', r'.*sign\s*up.*', r'.*register.*',
    r'.*create.*account.*', r'.*free.*account.*', r'.*subscription.*',
    r'.*paywall.*', r'.*access.*denied.*', r'.*get\s+.*access.*',
    r'.*premium.*content.*', r'.*unlock.*content.*', r'.*full.*access.*',
    r'.*purchase.*'
]

# Inline chrome filtering configuration
INLINE_SHORT_LINE_MAXLEN = 140
CHROME_INLINE_KEYWORDS = [
    'sign in', 'sign up', 'log in', 'log-in', 'register', 'subscribe',
    'forgot password', 'username', 'password', 'privacy', 'terms', 'cookies',
    'share on', 'facebook', 'twitter', 'linkedin', 'email'
]

# DOI extraction schema for XPath-based extraction from academic publishers
DOI_EXTRACTION_SCHEMA = {
    "name": "DOI extractor (XPath)",
    "baseSelector": "/html",
    "fields": [
        {
            "name": "doi_meta_cite",
            "selector": "//meta[@name='citation_doi']",
            "type": "attribute",
            "attribute": "content"
        },
        {
            "name": "doi_meta_pub",
            "selector": "//meta[@name='publication_doi']",
            "type": "attribute",
            "attribute": "content"
        },
        {
            "name": "doi_dc",
            "selector": "//meta[translate(@name,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='dc.identifier' and contains(@content, 'doi.org/')]",
            "type": "attribute",
            "attribute": "content"
        },
        {
            "name": "doi_dc_doi",
            "selector": "//meta[translate(@name,'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz')='dc.identifier.doi']",
            "type": "attribute",
            "attribute": "content"
        },
        {
            "name": "doi_canonical",
            "selector": "//link[@rel='canonical' and contains(@href, 'doi.org/')]",
            "type": "attribute",
            "attribute": "href"
        }
    ]
}

# =============================================================================
# Full Text Expansion (Crawl4AI JS Instrumentation)
# =============================================================================
# Clicks on common "Full Text" controls and waits for the page to settle so that
# dynamically loaded content becomes part of the DOM before extraction.
CLICK_AND_MONITOR_JS = r"""
(() => {
  if (!window.__c4ai_mon) {
    const mon = window.__c4ai_mon = {
      inflight: 0,
      last: Date.now(),
      started: false,
      mutations: 0,
      baseline: (document.body.innerText || '').length,
    };
    const bump = () => { mon.last = Date.now(); };

    const origFetch = window.fetch;
    if (origFetch) {
      window.fetch = (...args) => {
        mon.inflight++; bump();
        return origFetch(...args)
          .finally(() => { mon.inflight--; bump(); });
      };
    }

    const XS = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.send = function(...args) {
      mon.inflight++; bump();
      this.addEventListener('loadend', () => { mon.inflight--; bump(); }, { once: true });
      return XS.apply(this, args);
    };

    const mo = new MutationObserver(muts => {
      mon.mutations += muts.length; bump();
      for (const m of muts) {
        m.addedNodes && m.addedNodes.forEach(n => {
          const t = n.tagName && n.tagName.toLowerCase();
          if (t === 'img' || t === 'iframe' || t === 'video' || t === 'audio') {
            n.addEventListener('load', bump, true);
            n.addEventListener('error', bump, true);
          }
        });
      }
    });
    mo.observe(document, { subtree: true, childList: true, characterData: true, attributes: true });

    window.addEventListener('load', bump, true);
    window.addEventListener('error', bump, true);
  }

  const mon = window.__c4ai_mon;
  const LABEL = /full\s*text/i;

  const candidates = [
    ...document.querySelectorAll('h1,h2,h3,h4,h5,h6')
  ].
    filter(h => LABEL.test((h.innerText || '').trim())).
    flatMap(h => [h, ...h.querySelectorAll('button,[role="button"],a')]).
    concat(
      [...document.querySelectorAll('button,[role="button"],a,[aria-label]')]
        .filter(el => LABEL.test((el.innerText || el.getAttribute('aria-label') || '').trim()))
    );

  const seen = new Set(); const uniq = [];
  for (const el of candidates) { if (el && !seen.has(el)) { seen.add(el); uniq.push(el); } }

  const isClosed = (el) => el.getAttribute && el.hasAttribute('aria-expanded')
    ? el.getAttribute('aria-expanded') !== 'true'
    : true;

  const click = (el) => {
    el.scrollIntoView({ block: 'center' });
    el.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true }));
    el.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
    el.dispatchEvent(new PointerEvent('pointerup', { bubbles: true }));
    el.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
    el.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', code: 'Enter', bubbles: true }));
    el.dispatchEvent(new KeyboardEvent('keyup',   { key: 'Enter', code: 'Enter', bubbles: true }));
  };

  let didAny = false;
  for (const el of uniq) {
    const target = (/^h[1-6]$/i.test(el.tagName) ? (el.querySelector('button,[role="button"],a') || el) : el);
    if (target && isClosed(target) && getComputedStyle(target).display !== 'none') {
      click(target); didAny = true;
    }
  }
  if (didAny) mon.started = true;
})();
"""

WAIT_UNTIL_IDLE_JS = (
  "js:() => {" \
  "  const m = window.__c4ai_mon; if (!m) return true;" \
  "  const idle = m.inflight === 0 && (Date.now() - m.last) > 1000;" \
  "  if (!m.started) return idle;" \
  "  const grew = (document.body.innerText || '').length > m.baseline + 50;" \
  "  return idle && (m.mutations > 0 || grew);" \
  "}"
)

# Threshold for deciding when to retry with stealth/full-text expansion
STEALTH_RETRY_THRESHOLD = 3000  # characters of raw markdown

# =============================================================================
# Content Processing Helper Functions
# =============================================================================

def refine_article_content(markdown: str) -> str:
    """
    Trim a page's Markdown to the core scholarly article while removing site chrome.

    The function assumes ATX-style headings ("#", "##", …) and proceeds in phases:
    1) Parse and classify headings; 2) Optionally pre-trim a duplicated title/abstract
       preamble; 3) Identify a core article region; 4) Remove irrelevant sections before,
       within, and after the core while preserving the true title and body.

    Logic and behaviour
    - Heading detection: Headings are extracted with their level and character offsets and
      classified into:
        • Relevant: academic sections like abstract, summary, introduction, methods,
          results, discussion, conclusion, background, etc.
        • Irrelevant: site chrome (navigation, sharing, cookies, privacy), metadata blocks
          (affiliations, authors lists), and boilerplate (acknowledgements, funding, etc.).
        • Paywall: login/register/paywall cues.

    - Duplicate intro pre-trim (safe optional step):
        • Section-repeat heuristic:
            1) Take the first identified relevant section (e.g., "Abstract").
            2) If the same-named relevant section appears again as the second relevant match,
               find the second match's preceding larger heading (nearest heading with a lower
               H level, e.g., H1 before H2 Abstract).
            3) If the first relevant section starts before that preceding larger heading,
               trim everything up to that larger heading (drops duplicate summary blocks),
               then re-run the routine on the trimmed content.
        • Fallbacks:
            - If a duplicate H1 title appears and an Abstract occurs before the second title,
              trim to the second title and re-run.
            - Else, if two Abstract headings exist, trim to the second Abstract and re-run.

    - No headings: Return the original `markdown` unchanged.

    - No relevant sections found:
        • If paywall-like headings are present, return an empty string.
        • Otherwise, aggressively remove each irrelevant section (from its heading to the
          next same-or-higher-level heading or end-of-document), keeping the remainder.

    - Core region (when relevant sections exist):
        • core_start: Prefer the first Abstract/Summary/Introduction. If found, preserve a
          title heading immediately preceding it that has a larger visual rank (lower level
          number, e.g., H1 before H2 Abstract). Remove everything before that title, and
          also remove the content between the title and Abstract (preserving the newline
          after the title). If no such title exists, remove everything before the Abstract.
          If no Abstract/Summary/Introduction is present, set core_start to the first
          non-irrelevant, non-paywall heading (fallback: first relevant) and remove the
          prefix.
        • core_end: The first irrelevant heading after the last relevant section, or the end
          of the document if none exists.

    - Section removals:
        • Before core_start: Remove the entire prefix (except the preserved title case).
        • Within the core: Remove any irrelevant sections that begin between core_start and
          core_end, cutting each from its heading to the next same-or-higher-level heading,
          but never beyond core_end.
        • Reference-like anywhere: If academic structure is detected, remove reference-like
          sections (e.g., references/bibliography/citations) anywhere in the document,
          trimming to the next same-or-higher-level heading.
        • After core_end: Do not cut the entire tail. Remove only irrelevant sections
          individually (bounded to the next same-or-higher-level heading), preserving any
          later relevant sections (e.g., Appendix).

    Safeguards
    - Over-trim guard: If the page does not clearly look like an academic article and the
      reduction would be extreme (final length < 30% of input), fall back to the original
      `markdown` to avoid accidental information loss on portal-like pages.
    - Robustness: The pre-trim phase is wrapped in try/except and falls back to the standard
      flow on anomalies.

    Args:
        markdown: Raw Markdown content from a candidate scholarly article page.

    Returns:
        Refined Markdown focused on the article title and core academic sections, with
        navigation, boilerplate, and other irrelevant blocks removed.

    Examples
        • Duplicate title + abstract: A top summary (Title + Abstract) followed by the main
          article is trimmed to the second Title and then processed as a normal article.
        • References mid-document: A mid-document "## References" is removed while keeping
          subsequent core content (e.g., "## Appendix").
    """

    # Use module-level constants
    relevant_headings = RELEVANT_HEADINGS
    irrelevant_headings = IRRELEVANT_HEADINGS
    irrelevant_patterns = IRRELEVANT_PATTERNS
    paywall_patterns = PAYWALL_PATTERNS
    all_irrel_patterns = irrelevant_patterns + paywall_patterns

    # Find all headings with their positions and levels
    heading_pattern = r'^(#{1,6})\s+(.+?)(?:\s*\{[^}]*\})?\s*$'
    headings = []

    for match in re.finditer(heading_pattern, markdown, re.MULTILINE):
        level = len(match.group(1))  # Number of # characters
        text = match.group(2).strip()
        start_pos = match.start()
        end_pos = match.end()

        # Classify heading as relevant or irrelevant
        text_lower = text.lower()

        # Check for irrelevant patterns first (higher priority)
        is_irrelevant = (
            any(irrel in text_lower for irrel in irrelevant_headings) or
            any(re.match(pattern, text_lower, re.IGNORECASE) for pattern in all_irrel_patterns)
        )

        # Check for paywall patterns
        is_paywall = any(re.match(pattern, text_lower, re.IGNORECASE) for pattern in paywall_patterns)

        # Only check for relevant patterns if not already marked as irrelevant
        is_relevant = not is_irrelevant and any(rel in text_lower for rel in relevant_headings)

        headings.append({
            'text': text,
            'level': level,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'is_relevant': is_relevant,
            'is_irrelevant': is_irrelevant,
            'is_paywall': is_paywall,
            'full_match': match.group(0)
        })

    if not headings:
        return markdown

    # Pre-pass: Detect duplicated intro (same relevant section repeated) and trim the first occurrence
    def _norm_title(s: str) -> str:
        s = re.sub(r"[\W_]+", " ", s or "").strip().lower()
        return re.sub(r"\s+", " ", s)

    try:
        # Heuristic: If the first relevant section appears twice and the second occurrence
        # has a preceding larger (lower H number) heading, and that larger heading appears
        # after the first relevant section, then trim up to that larger heading.
        rels = [h for h in headings if h['is_relevant']]
        if len(rels) >= 2:
            def _norm_section_name(txt: str) -> str:
                return _norm_title(txt)

            first_rel = rels[0]
            first_rel_name = _norm_section_name(first_rel['text'])

            second_rel_same = None
            for r in rels[1:]:
                if _norm_section_name(r['text']) == first_rel_name:
                    second_rel_same = r
                    break

            def _preceding_larger_heading(target) -> dict | None:
                for h in reversed(headings):
                    if h['start_pos'] < target['start_pos'] and h['level'] < target['level']:
                        return h
                return None

            if second_rel_same is not None:
                parent2 = _preceding_larger_heading(second_rel_same)
                if parent2 is not None and first_rel['start_pos'] < parent2['start_pos']:
                    cut_pos = parent2['start_pos']
                    # Never trim past first recognized heading
                    if first_relevant_pos is None or cut_pos <= first_relevant_pos:
                        new_markdown = markdown[cut_pos:]
                        if len(new_markdown) < len(markdown):
                            return refine_article_content(new_markdown)

        h1s = [h for h in headings if h['level'] == 1]
        abstracts = [h for h in headings if 'abstract' in h['text'].lower()]
        trimmed_prefix = False

        if len(h1s) >= 2:
            first_title = _norm_title(h1s[0]['text'])
            for h in h1s[1:]:
                if _norm_title(h['text']) == first_title:
                    cut_pos = h['start_pos']
                    # Only trim if an Abstract appears before the duplicate title (indicating a summary header)
                    if abstracts and abstracts[0]['start_pos'] < cut_pos:
                        # Never trim past first recognized heading
                        if first_relevant_pos is None or cut_pos <= first_relevant_pos:
                            new_markdown = markdown[cut_pos:]
                            if len(new_markdown) < len(markdown):
                                return refine_article_content(new_markdown)
                    break

        # If no duplicate title found but multiple Abstracts exist, trim to second Abstract
        if not trimmed_prefix and len(abstracts) >= 2:
            cut_pos = abstracts[1]['start_pos']
            # Never trim past first recognized heading
            if first_relevant_pos is None or cut_pos <= first_relevant_pos:
                new_markdown = markdown[cut_pos:]
                if len(new_markdown) < len(markdown):
                    return refine_article_content(new_markdown)
    except Exception:
        # If any issue occurs in pre-pass, continue with normal processing
        pass

    # Find first and last relevant headings (be robust: also match by keyword even if flags are off)
    relevant_headings_list = [h for h in headings if h['is_relevant']]
    keyword_relevant_list = [h for h in headings if any(rel in h['text'].lower() for rel in RELEVANT_HEADINGS)]
    irrelevant_headings_list = [h for h in headings if h['is_irrelevant']]
    paywall_headings_list = [h for h in headings if h['is_paywall']]

    # Build list of sections to remove
    sections_to_remove = []

    if not relevant_headings_list:
        # No relevant headings found — preserve the primary content band:
        # - Start at the first non-irrelevant, non-paywall heading (prefer lower H levels)
        # - Remove everything before that heading (site chrome)
        # - Then remove from the first irrelevant heading AFTER that content heading to the end

        if headings:
            # Compute section spans for each heading
            spans = []
            for i, h in enumerate(headings):
                sec_start = h['end_pos']
                sec_end = headings[i+1]['start_pos'] if i+1 < len(headings) else len(markdown)
                sec_len = len(markdown[sec_start:sec_end].strip())
                spans.append((h, sec_start, sec_end, sec_len))

            # Prefer a content heading (not irrelevant/paywall, level<=3) with substantial section length
            def pick_content_head(min_len: int = 200):
                cands = [h for (h, a, b, L) in spans if not h['is_irrelevant'] and not h['is_paywall'] and h['level'] <= 3 and L >= min_len]
                if cands:
                    return sorted(cands, key=lambda h: (h['level'], h['start_pos']))[0]
                cands = [h for (h, a, b, L) in spans if not h['is_irrelevant'] and not h['is_paywall'] and L >= min_len]
                if cands:
                    return sorted(cands, key=lambda h: (h['level'], h['start_pos']))[0]
                # Fallback: first non-irrelevant/paywall heading, then first heading
                cands = [h for (h, a, b, L) in spans if not h['is_irrelevant'] and not h['is_paywall']]
                if cands:
                    return sorted(cands, key=lambda h: (h['level'], h['start_pos']))[0]
                return min(headings, key=lambda h: h['start_pos'])

            content_head = pick_content_head()

            # Remove chrome before primary content heading
            if content_head['start_pos'] > 0:
                sections_to_remove.append((0, content_head['start_pos']))

            # Remove every irrelevant section after the content heading, bounded to next
            for h in headings:
                if h['is_irrelevant'] and h['start_pos'] > content_head['start_pos']:
                    section_end = len(markdown)
                    for nh in headings:
                        if nh['start_pos'] > h['start_pos'] and nh['level'] <= h['level']:
                            section_end = nh['start_pos']
                            break
                    sections_to_remove.append((h['start_pos'], section_end))
    else:
        # Use keyword-based fallback to determine the last relevant heading reliably
        first_relevant = relevant_headings_list[0]
        last_relevant = (keyword_relevant_list[-1] if keyword_relevant_list else relevant_headings_list[-1])

        # Look for Abstract, Summary, or Introduction section first
        abstract_heading = None
        for heading in relevant_headings_list:
            text_lower = heading['text'].lower()
            if ('abstract' in text_lower) or ('summary' in text_lower) or ('introduction' in text_lower):
                abstract_heading = heading
                break

        # Determine a core_start and preserve title if present
        core_start = first_relevant['start_pos']
        title_to_preserve = None
        if abstract_heading is not None:
            core_start = abstract_heading['start_pos']
            # Look for a larger heading immediately before Abstract and preserve it as title
            for heading in reversed(headings):
                if (heading['start_pos'] < abstract_heading['start_pos'] and heading['level'] < abstract_heading['level']):
                    # Always preserve the title-level heading immediately before Abstract,
                    # regardless of intervening same/higher-level headings. This prevents
                    # accidental removal of the true article title.
                    title_to_preserve = heading
                    break

            if title_to_preserve is not None:
                # Remove content before title
                sections_to_remove.append((0, title_to_preserve['start_pos']))
                # Remove content between title and Abstract, but preserve the newline after title
                title_line_end = title_to_preserve['end_pos']
                while title_line_end < len(markdown) and markdown[title_line_end] != '\n':
                    title_line_end += 1
                if title_line_end < len(markdown):
                    title_line_end += 1
                sections_to_remove.append((title_line_end, abstract_heading['start_pos']))
            else:
                # No title found, remove everything before Abstract
                sections_to_remove.append((0, abstract_heading['start_pos']))
        else:
            # No Abstract/Summary/Introduction section - start at the first non-irrelevant heading
            first_academic_heading = None
            for heading in headings:
                if (not heading['is_irrelevant'] and not heading['is_paywall']):
                    first_academic_heading = heading
                    break
            core_start = (first_academic_heading or first_relevant)['start_pos']
            sections_to_remove.append((0, core_start))

        # Determine core_end as first irrelevant heading after the last relevant section,
        # otherwise end of document. Then, ensure we don't cut off any later relevant section
        # by mistake (robustness against misclassification).
        core_end = len(markdown)
        for heading in headings:
            if heading['start_pos'] > last_relevant['start_pos'] and heading['is_irrelevant']:
                core_end = heading['start_pos']
                break
        # If any relevant-looking heading appears at or after core_end, extend core_end to end
        if core_end < len(markdown):
            for h in headings:
                if h['start_pos'] >= core_end and (h.get('is_relevant') or any(rel in h['text'].lower() for rel in RELEVANT_HEADINGS)):
                    core_end = len(markdown)
                    break

        # Remove any irrelevant sections within the core region (clipped)
        for heading in headings:
            if heading['is_irrelevant'] and core_start <= heading['start_pos'] < core_end:
                # End at next heading of same/higher level, but don't cross core_end
                section_end = core_end
                for next_heading in headings:
                    if next_heading['start_pos'] > heading['start_pos'] and next_heading['level'] <= heading['level']:
                        section_end = min(section_end, next_heading['start_pos'])
                        break
                sections_to_remove.append((heading['start_pos'], section_end))

        # Also remove irrelevant sections that start after core_end individually,
        # rather than cutting the entire tail. This avoids removing late relevant content
        # (e.g., Appendix) if present.
        if core_end < len(markdown):
            for heading in headings:
                if heading['is_irrelevant'] and heading['start_pos'] >= core_end:
                    section_end = len(markdown)
                    for next_heading in headings:
                        if next_heading['start_pos'] > heading['start_pos'] and next_heading['level'] <= heading['level']:
                            section_end = next_heading['start_pos']
                            break
                    sections_to_remove.append((heading['start_pos'], section_end))

        # Additionally, remove reference-like sections anywhere in the document,
        # but only if we detected academic structure. This avoids over-removal
        # on navigation-heavy portal pages.
        if relevant_headings_list:
            for heading in headings:
                text_lower = heading['text'].lower()
                is_reference_like = (
                    'references' in text_lower or 'bibliography' in text_lower or 'citations' in text_lower
                )
                # Only add extra removals for reference-like headings that weren't already
                # flagged as irrelevant to avoid duplicate intervals.
                if is_reference_like and not heading['is_irrelevant']:
                    # Find end of this section: next heading of same or higher level, or end of doc
                    section_end = len(markdown)
                    for next_heading in headings:
                        if next_heading['start_pos'] > heading['start_pos'] and next_heading['level'] <= heading['level']:
                            section_end = next_heading['start_pos']
                            break
                    sections_to_remove.append((heading['start_pos'], section_end))

    # Apply removals in reverse order to maintain position accuracy
    sections_to_remove.sort(key=lambda x: x[0], reverse=True)
    result = markdown

    for start_pos, end_pos in sections_to_remove:
        result = result[:start_pos] + result[end_pos:]

    # Safety check: only apply when we don't have clear academic structure
    # For content with clear relevant sections and many irrelevant sections, aggressive removal is desired
    has_clear_academic_structure = (
        len(relevant_headings_list) > 0 and
        len(irrelevant_headings_list) > 5 and
        any((('abstract' in h['text'].lower()) or ('summary' in h['text'].lower()) or ('introduction' in h['text'].lower())) for h in relevant_headings_list)
    )

    # Alternative academic structure: papers without Abstract but with standard sections
    has_alternative_academic_structure = (
        len(relevant_headings_list) >= 3 and  # At least 3 relevant sections
        len(irrelevant_headings_list) > 5 and  # Many irrelevant sections to remove
        any(section in h['text'].lower() for h in relevant_headings_list
            for section in ['methods', 'results', 'discussion', 'conclusion'])
    )

    # Inline chrome remover: drop short nav/policy lines and irrelevant sections missed by structure pass
    def _cleanup_inline(text: str) -> str:
        lines = text.splitlines()
        cleaned = []
        skip_until_next_heading = False
        heading_re = re.compile(r'^(#{1,6})\s+(.+?)\s*$')
        for line in lines:
            m = heading_re.match(line)
            if m:
                skip_until_next_heading = False
                htxt = m.group(2).strip().lower()
                # If heading itself is irrelevant, skip this heading and the section content lines
                if (any(h in htxt for h in IRRELEVANT_HEADINGS)
                    or any(re.match(p, htxt, re.IGNORECASE) for p in IRRELEVANT_PATTERNS)
                    or any(re.match(p, htxt, re.IGNORECASE) for p in PAYWALL_PATTERNS)):
                    skip_until_next_heading = True
                    continue
                cleaned.append(line)
                continue
            if skip_until_next_heading:
                continue
            s = line.strip()
            # Drop obvious short chrome lines
            if s and len(s) <= INLINE_SHORT_LINE_MAXLEN:
                low = s.lower()
                if any(k in low for k in CHROME_INLINE_KEYWORDS):
                    continue
            cleaned.append(line)
        return "\n".join(cleaned).strip()

    # Skip safety check for clear academic content (e.g., abstract pages with lots of website chrome)
    if not has_clear_academic_structure and not has_alternative_academic_structure:
        orig = markdown.strip()
        refined = result.strip()
        # If refinement removed everything on a non-academic page, keep original
        if orig and not refined:
            return _cleanup_inline(orig)
        # If refinement is extremely aggressive, keep original
        if orig and refined:
            reduction_ratio = len(refined) / len(orig)
            if reduction_ratio < 0.3:
                return _cleanup_inline(orig)

    refined = result.strip()
    if not refined and headings:
        # Final fallback for non-academic pages: keep band from first heading
        first_heading = min(headings, key=lambda h: h['start_pos'])
        end = len(markdown)
        for h in headings:
            if h['start_pos'] > first_heading['start_pos'] and h['is_irrelevant']:
                end = h['start_pos']
                break
        band = markdown[first_heading['start_pos']:end].strip()
        if band:
            return _cleanup_inline(band)
    return _cleanup_inline(refined)

def extract_headings(markdown: str) -> List[dict]:
    """
    Extract all headings from markdown content with metadata.

    Args:
        markdown: Markdown content to analyze

    Returns:
        List of heading dictionaries with text, level, and position information
    """
    heading_pattern = r'^(#{1,6})\s+(.+?)(?:\s*\{[^}]*\})?\s*$'
    headings = []

    for match in re.finditer(heading_pattern, markdown, re.MULTILINE):
        level = len(match.group(1))
        text = match.group(2).strip()
        start_pos = match.start()
        end_pos = match.end()

        headings.append({
            'text': text,
            'level': level,
            'start_pos': start_pos,
            'end_pos': end_pos,
            'full_match': match.group(0)
        })

    return headings

def classify_heading_relevance(heading_text: str) -> dict:
    """
    Classify a heading as relevant, irrelevant, or paywall content.

    Args:
        heading_text: The text content of the heading

    Returns:
        Dictionary with 'is_relevant', 'is_irrelevant', 'is_paywall' boolean flags
    """
    # Use module-level constants
    relevant_headings = RELEVANT_HEADINGS
    irrelevant_headings = IRRELEVANT_HEADINGS
    irrelevant_patterns = IRRELEVANT_PATTERNS
    paywall_patterns = PAYWALL_PATTERNS

    text_lower = heading_text.lower()

    is_irrelevant = (
        any(irrel in text_lower for irrel in irrelevant_headings) or
        any(re.match(pattern, text_lower, re.IGNORECASE) for pattern in all_irrel_patterns)
    )

    is_paywall = any(re.match(pattern, text_lower, re.IGNORECASE) for pattern in paywall_patterns)
    is_relevant = not is_irrelevant and any(rel in text_lower for rel in relevant_headings)

    return {
        'is_relevant': is_relevant,
        'is_irrelevant': is_irrelevant,
        'is_paywall': is_paywall
    }

# =============================================================================
# Lazy Import Cache Variables
# =============================================================================

_crawl4ai_html_imports = None
_crawl4ai_pdf_imports = None
_quiet_logger = None
_chunker = None

def _get_crawl4ai_imports():
    """Lazy import crawl4ai modules with simple caching."""
    global _crawl4ai_html_imports
    if _crawl4ai_html_imports is None:
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
            from crawl4ai.extraction_strategy import JsonXPathExtractionStrategy
            from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
            from crawl4ai.async_logger import AsyncLoggerBase
            _crawl4ai_html_imports = (AsyncWebCrawler, CrawlerRunConfig, JsonXPathExtractionStrategy, AsyncLoggerBase, DefaultMarkdownGenerator)
        except ImportError:
            raise RuntimeError("crawl4ai package is required for fetching. Install with: pip install crawl4ai")
    return _crawl4ai_html_imports

def _get_crawl4ai_pdf_imports():
    """Lazy import crawl4ai PDF modules with simple caching."""
    global _crawl4ai_pdf_imports
    if _crawl4ai_pdf_imports is None:
        try:
            from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
            from crawl4ai.processors.pdf import PDFCrawlerStrategy, PDFContentScrapingStrategy
            from crawl4ai.async_logger import AsyncLoggerBase
            _crawl4ai_pdf_imports = (AsyncWebCrawler, CrawlerRunConfig, PDFCrawlerStrategy, PDFContentScrapingStrategy, AsyncLoggerBase)
        except ImportError:
            raise RuntimeError("crawl4ai package is required for fetching. Install with: pip install crawl4ai")
    return _crawl4ai_pdf_imports

def _get_browser_config(*, enable_stealth: bool = False, headless: bool = True):
    """Create and return a BrowserConfig instance with desired options."""
    try:
        from crawl4ai import BrowserConfig
    except ImportError:
        raise RuntimeError("crawl4ai package is required for fetching. Install with: pip install crawl4ai")
    return BrowserConfig(enable_stealth=enable_stealth, headless=headless)

def _build_html_crawler_config(timeout_ms: int, *, with_js: bool) -> "CrawlerRunConfig":
    """Factory for CrawlerRunConfig for HTML pages, sharing common options."""
    # Full tuple unpack to detect signature changes
    AsyncWebCrawler, CrawlerRunConfig, JsonXPathExtractionStrategy, AsyncLoggerBase, DefaultMarkdownGenerator = _get_crawl4ai_imports()
    md_gen = DefaultMarkdownGenerator(
        content_source="cleaned_html",
        options={
            "ignore_links": True,
            "ignore_images": True,
            "escape_html": True,
            "skip_internal_links": True,
        },
    )
    extra_kwargs = {}
    if with_js:
        extra_kwargs = {
            "js_code": [CLICK_AND_MONITOR_JS],
            "wait_for": WAIT_UNTIL_IDLE_JS,
            "delay_before_return_html": 2.0,
        }
    return CrawlerRunConfig(
        extraction_strategy=JsonXPathExtractionStrategy(DOI_EXTRACTION_SCHEMA, verbose=False),
        page_timeout=timeout_ms,
        word_count_threshold=10,
        excluded_tags=["nav", "footer", "aside", "form", "dialog"],
        excluded_selector="[role=dialog], .footer",
        verbose=False,
        markdown_generator=md_gen,
        **extra_kwargs,
    )

def _has_recognized_sections(markdown: str) -> bool:
    """Detect if markdown contains recognized academic sections based on headings."""
    try:
        heads = extract_headings(markdown)
        for h in heads:
            cls = classify_heading_relevance(h['text'])
            if cls.get('is_relevant'):
                return True
    except Exception:
        pass
    return False

def _should_retry_with_stealth(raw_markdown: str, raw_html: str) -> tuple[bool, str]:
    """Decide whether to retry fetching with stealth/full-text instrumentation.

    Returns:
        (should_retry, reason)
    """
    reasons: list[str] = []
    text = ((raw_markdown or "") + "\n" + (raw_html or "")).lower()

    if not raw_markdown:
        reasons.append("no_markdown")
    else:
        if len(raw_markdown) < STEALTH_RETRY_THRESHOLD:
            reasons.append("short_markdown")

    recognized = _has_recognized_sections(raw_markdown or "")
    if not recognized:
        reasons.append("no_recognized_sections")

    if "verifying you are human" in text:
        reasons.append("bot_challenge")

    # Retry if we have a bot challenge OR content is short/lacking sections
    should_retry = ("bot_challenge" in reasons) or (
        ("short_markdown" in reasons or "no_markdown" in reasons) and ("no_recognized_sections" in reasons)
    )

    return should_retry, ",".join(reasons) if reasons else "sufficient_content"

def _get_granular_logger(status_display=None):
    """
    Create a logger that provides granular stage information to Rich status displays.

    Args:
        status_display: Rich status object to update with progress information

    Returns:
        GranularLogger instance that updates the status display with crawl4ai stages
    """
    # Full tuple unpack so signature changes raise immediately
    AsyncWebCrawler, CrawlerRunConfig, JsonXPathExtractionStrategy, AsyncLoggerBase, DefaultMarkdownGenerator = _get_crawl4ai_imports()

    class GranularLogger(AsyncLoggerBase):
        """Logger that translates crawl4ai stages into Rich status updates."""

        def __init__(self, status_display=None):
            self.status_display = status_display
        def debug(self, message: str, tag: str = "DEBUG", **kwargs): pass
        def info(self, message: str, tag: str = "INFO", **kwargs): pass
        def success(self, message: str, tag: str = "SUCCESS", **kwargs): pass
        def warning(self, message: str, tag: str = "WARNING", **kwargs): pass
        def error(self, message: str, tag: str = "ERROR", **kwargs): pass
        def error_status(self, url: str, error: str, tag: str = "ERROR", url_length: int = 100): pass

        def url_status(self, url: str, success: bool, timing: float,
                       tag: str = "FETCH", url_length: int = 100) -> None:
            if self.status_display is None:
                return

            domain = url.split('//')[1].split('/')[0] if '//' in url else url[:20]

            stage_messages = {
                "FETCH": f"[cyan]Fetching from[/cyan] [bold]{domain}[/bold]",
                "SCRAPE": f"[blue]Processing content from[/blue] [bold]{domain}[/bold]",
                "EXTRACT": f"[yellow]Extracting data from[/yellow] [bold]{domain}[/bold]",
                "COMPLETE": f"[green]✓ Completed[/green] [bold]{domain}[/bold] [dim]({timing:.1f}s)[/dim]"
            }

            if tag in stage_messages and hasattr(self.status_display, 'update'):
                self.status_display.update(stage_messages[tag])

    return GranularLogger(status_display)

def _get_chunker():
    """Lazy import and initialize chonkie chunker with simple caching."""
    global _chunker
    if _chunker is None:
        from chonkie import SDPMChunker
        _chunker = SDPMChunker(
            embedding_model="minishlab/potion-base-8M", # Default model
            threshold=0.5,                              # Similarity threshold (0-1)
            chunk_size=4096,                            # Maximum tokens per chunk
            min_sentences=2,                            # Initial sentences per chunk
            skip_window=1                               # Number of chunks to skip when looking for similarities
        )
    return _chunker

def _get_quiet_logger():
    """Get a silent logger for crawl4ai when no status updates are needed."""
    global _quiet_logger
    if _quiet_logger is None:
        _quiet_logger = _get_granular_logger(status_display=None)
    return _quiet_logger

class StatusDisplay:
    """Manages Rich status indicators for fetch operations."""

    def __init__(self, show_status: bool = True):
        self.show_status = show_status
        self.console = Console() if show_status else None
        # Avoid noisy "heartbeat" updates when not attached to an interactive TTY
        if self.console and not self.console.is_terminal:
            self.show_status = False

    def create_status(self, message: str, progress_info=None):
        """
        Create appropriate status indicator based on context.

        Args:
            message: Initial status message to display
            progress_info: Optional tuple of (progress_instance, task_id) for progress bars

        Returns:
            Rich status object or DummyStatus for no-op
        """
        if progress_info is not None:
            progress_instance, task_id = progress_info
            progress_instance.update(task_id, description=message)
            return DummyStatus()
        elif self.show_status and self.console:
            return self.console.status(message, spinner="dots")
        else:
            return DummyStatus()

class DummyStatus:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def update(self, text: str):
        pass
def url_to_hash(url: str) -> str:
    """Convert URL to a filesystem-safe hash (backward compatibility)."""
    normalized_url = normalize_url(url)
    return hashlib.sha256(normalized_url.encode('utf-8')).hexdigest()[:16]

def normalize_url(url: str) -> str:
    """Normalize URL by removing fragment and other client-side only components."""
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    # Remove fragment (everything after #) as it's client-side only
    return urlunparse(parsed._replace(fragment=''))

def url_to_hash_base36(url: str) -> str:
    """Convert URL to a base36 hash for compact representation."""
    normalized_url = normalize_url(url)
    # Use deterministic SHA256 hash instead of Python's non-deterministic hash()
    hash_bytes = hashlib.sha256(normalized_url.encode('utf-8')).digest()
    hash_int = int.from_bytes(hash_bytes[:8], byteorder='big')  # Use first 8 bytes

    # Manual base36 conversion
    if hash_int == 0:
        return "0"

    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = ""
    while hash_int:
        result = digits[hash_int % 36] + result
        hash_int //= 36
    return result

class URLCache:
    """File-based cache for URLs with support for HTML, PDF, and Markdown content."""

    def __init__(self, config: IfetcherConfig):
        self.config = config
        self.base_path = config.abspath(config.output.cache)
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def _get_paths(self, url: str, *extensions: str) -> tuple[Path, ...]:
        """Get file paths for specific extensions with collision resolution."""
        normalized_url = normalize_url(url)
        base_hash = url_to_hash_base36(url)
        probe = 0

        while probe < 1000:  # Safety limit
            if probe == 0:
                hash_str = base_hash
            else:
                hash_str = f"{base_hash}_{probe}"

            url_path = self.base_path / f"{hash_str}.url"

            # If no URL file exists, this slot is available
            if not url_path.exists():
                return tuple(self.base_path / f"{hash_str}.{ext}" for ext in extensions)

            # URL file exists, check if it's for the same URL
            try:
                async with aiofiles.open(url_path, 'r', encoding='utf-8') as f:
                    stored_url = (await f.read()).strip()
                if stored_url == normalized_url:
                    # Found existing entry for this URL
                    return tuple(self.base_path / f"{hash_str}.{ext}" for ext in extensions)
            except (OSError, UnicodeDecodeError):
                # Corrupted file, skip this slot
                pass

            probe += 1

        raise RuntimeError(f"Too many hash collisions for URL: {url}")

    async def _store_url_mapping(self, url: str, final_url: Optional[str] = None) -> None:
        """Store the URL mapping and redirect info in sidecar files with atomic operations."""
        normalized_url = normalize_url(url)
        url_path, redir_path = await self._get_paths(url, "url", "redir")

        # Atomic file creation: write to temp file then rename
        temp_url_path = url_path.with_suffix('.url.tmp')
        try:
            async with aiofiles.open(temp_url_path, 'w', encoding='utf-8') as f:
                await f.write(normalized_url)
            # Atomic rename - prevents race conditions
            await aiofiles.os.rename(temp_url_path, url_path)
        except Exception:
            # Clean up temp file on failure
            if temp_url_path.exists():
                temp_url_path.unlink()
            raise

        # Store redirect mapping if final URL differs from original
        if final_url and normalize_url(final_url) != normalized_url:
            temp_redir_path = redir_path.with_suffix('.redir.tmp')
            try:
                async with aiofiles.open(temp_redir_path, 'w', encoding='utf-8') as f:
                    await f.write(normalize_url(final_url))
                await aiofiles.os.rename(temp_redir_path, redir_path)
            except Exception:
                if temp_redir_path.exists():
                    temp_redir_path.unlink()
                raise
        elif redir_path.exists():
            # Remove stale redirect file if URLs now match
            redir_path.unlink()

    async def _verify_url_mapping(self, url: str) -> bool:
        """Verify the URL mapping matches what's stored."""
        normalized_url = normalize_url(url)
        url_path, = await self._get_paths(url, "url")
        if url_path.exists():
            try:
                async with aiofiles.open(url_path, 'r', encoding='utf-8') as f:
                    stored_url = (await f.read()).strip()
                return stored_url == normalized_url
            except (OSError, UnicodeDecodeError):
                return False
        return False

    async def has_path(self, url: str, content_type: str) -> bool:
        """Check if content with given content type is cached for URL (including redirected URLs)."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(f"Unknown content type '{content_type}'. Available: {available}")

        extension = CONTENT_TYPE_CONFIG[content_type]["extension"]
        content_path, redir_path = await self._get_paths(url, extension, "redir")
        if content_path.exists():
            if await self._verify_url_mapping(url):
                return True
        # Check if this URL redirected to another URL that has content
        if redir_path.exists():
            try:
                async with aiofiles.open(redir_path, 'r', encoding='utf-8') as f:
                    final_url = (await f.read()).strip()
                final_content_path, = await self._get_paths(final_url, extension)
                return final_content_path.exists()
            except (OSError, UnicodeDecodeError):
                pass
        return False

    async def get_path(self, url: str, content_type: str) -> str:
        """Get cached content for URL with given content type (following redirects if needed)."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(f"Unknown content type '{content_type}'. Available: {available}")

        extension = CONTENT_TYPE_CONFIG[content_type]["extension"]
        content_path, redir_path = await self._get_paths(url, extension, "redir")

        # Check if content exists
        if content_path.exists():
            if await self._verify_url_mapping(url):
                async with aiofiles.open(content_path, 'r', encoding='utf-8') as f:
                    return await f.read()

        # Check for redirected content
        if redir_path.exists():
            try:
                async with aiofiles.open(redir_path, 'r', encoding='utf-8') as f:
                    final_url = (await f.read()).strip()
                final_content_path, = await self._get_paths(final_url, extension)
                if final_content_path.exists():
                    async with aiofiles.open(final_content_path, 'r', encoding='utf-8') as f:
                        return await f.read()
            except (OSError, UnicodeDecodeError):
                pass

        raise KeyError(f"Content with extension '{extension}' not cached for URL: {url}")

    async def set_path(self, url: str, content_type: str, content: str, final_url: Optional[str] = None) -> None:
        """Store content for URL and content type with atomic operations."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(f"Unknown content type '{content_type}'. Available: {available}")

        extension = CONTENT_TYPE_CONFIG[content_type]["extension"]
        # If there's a redirect, store content at the final URL location
        storage_url = final_url if final_url else url
        path, = await self._get_paths(storage_url, extension)

        # Atomic content write
        temp_path = path.with_suffix(f'.{extension}.tmp')
        try:
            async with aiofiles.open(temp_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            await aiofiles.os.rename(temp_path, path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        # Store URL mapping for both original and final URLs
        await self._store_url_mapping(url, final_url)
        if final_url and final_url != url:
            # Also store mapping for final URL to itself (for direct access)
            await self._store_url_mapping(final_url, None)

    async def get_content(self, url: str, content_type: str):
        """Get cached content for URL with given type (following redirects if needed)."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(f"Unknown content type '{content_type}'. Available: {available}")

        if content_type == "doi":
            try:
                raw_content = await self.get_path(url, content_type)
                return CONTENT_TYPE_CONFIG[content_type]["deserialize"](raw_content)
            except KeyError:
                return None
        else:
            raw_content = await self.get_path(url, content_type)
            return CONTENT_TYPE_CONFIG[content_type]["deserialize"](raw_content)

    async def set_content(self, url: str, content_type: str, content, final_url: Optional[str] = None) -> None:
        """Store content for URL with given type."""
        if content_type not in CONTENT_TYPE_CONFIG:
            available = ", ".join(CONTENT_TYPE_CONFIG.keys())
            raise ValueError(f"Unknown content type '{content_type}'. Available: {available}")

        serialized_content = CONTENT_TYPE_CONFIG[content_type]["serialize"](content)
        await self.set_path(url, content_type, serialized_content, final_url)

    def _get_extension(self, content_type: str) -> str:
        """Get file extension for a content type."""
        return CONTENT_TYPE_CONFIG[content_type]["extension"]

    async def has_url(self, url: str) -> bool:
        """Check if URL is cached in any format."""
        for content_type in CONTENT_TYPE_CONFIG.keys():
            if await self.has_path(url, content_type):
                return True
        return False

    async def get_source_type(self, url: str) -> Optional[str]:
        """Get the source content type for a URL ('html' or 'pdf')."""
        if await self.has_path(url, "html"):
            return "html"
        elif await self.has_path(url, "pdf"):
            return "pdf"
        return None

    async def get_source_content(self, url: str) -> Optional[str]:
        """Get the raw source content (HTML or PDF) for a URL."""
        if await self.has_path(url, "html"):
            return await self.get_content(url, "html")
        elif await self.has_path(url, "pdf"):
            return await self.get_content(url, "pdf")
        return None

    async def clear_url(self, url: str) -> None:
        """Remove all cached content for a URL."""
        extensions = [config["extension"] for config in CONTENT_TYPE_CONFIG.values()]
        extensions.extend(["url", "redir", "failed"])  # Add metadata file extensions
        paths = await self._get_paths(url, *extensions)
        for path in paths:
            if path.exists():
                path.unlink()

    async def get_url_hash(self, url: str) -> str:
        """Get the actual hash string used for a URL (including probe suffix if any)."""
        url_path, = await self._get_paths(url, "url")
        # Extract hash from the path name
        return url_path.stem  # Remove .url extension to get the hash

    async def get_original_url(self, hash_str: str) -> Optional[str]:
        """Get the original URL from a hash string."""
        url_path = self.base_path / f"{hash_str}.url"
        if url_path.exists():
            try:
                async with aiofiles.open(url_path, 'r', encoding='utf-8') as f:
                    return (await f.read()).strip()
            except (OSError, UnicodeDecodeError):
                return None
        return None

    async def get_redirect_info(self, url: str) -> Optional[str]:
        """Get the final URL if this URL redirected, None otherwise."""
        redir_path, = await self._get_paths(url, "redir")
        if redir_path.exists():
            try:
                async with aiofiles.open(redir_path, 'r', encoding='utf-8') as f:
                    content = (await f.read()).strip()
                # Validate that the content looks like a URL
                if content and '://' in content:
                    return content
            except (OSError, UnicodeDecodeError):
                pass
        return None

    async def get_failed_reason(self, url: str) -> Optional[str]:
        """Read and return the stored failure reason for this exact URL if present."""
        failed_path, = await self._get_paths(url, "failed")
        if failed_path.exists():
            try:
                async with aiofiles.open(failed_path, 'r', encoding='utf-8') as f:
                    return (await f.read()).strip()
            except (OSError, UnicodeDecodeError):
                return None
        return None

    async def is_failed(self, url: str) -> bool:
        """Check if a previous failure sentinel exists for this exact URL.

        Separation of concerns: redirect resolution should be handled by the
        caller (e.g., PageFetcher), which can then call `is_failed` on the
        resolved URL if desired.
        """
        failed_path, = await self._get_paths(url, "failed")
        return failed_path.exists()

    async def mark_failed(self, url: str, final_url: Optional[str] = None, reason: Optional[str] = None) -> None:
        """Create/overwrite a .failed sentinel for this URL.

        Writes the plain-text error message to `<hash>.failed` for the provided URL.
        We do not duplicate writes for redirect targets; inheritance is handled at
        read-time by `is_failed()` following the `.redir` chain.
        """
        content = (reason or "").strip()
        path, = await self._get_paths(url, "failed")
        tmp = path.with_suffix('.failed.tmp')
        try:
            async with aiofiles.open(tmp, 'w', encoding='utf-8') as f:
                await f.write(content)
            await aiofiles.os.rename(tmp, path)
        except Exception:
            if tmp.exists():
                tmp.unlink()
            raise

    async def clear_failed(self, url: str) -> None:
        """Remove any existing .failed sentinel for this exact URL only."""
        failed_path, = await self._get_paths(url, "failed")
        if failed_path.exists():
            failed_path.unlink()

    async def list_cached_urls(self) -> list[str]:
        """Get a list of all cached URLs."""
        urls = []
        for url_file in self.base_path.glob("*.url"):
            try:
                async with aiofiles.open(url_file, 'r', encoding='utf-8') as f:
                    url = (await f.read()).strip()
                # Verify at least one content file exists
                if await self.has_url(url):
                    urls.append(url)
            except (OSError, UnicodeDecodeError):
                continue
        return urls

class PageFetcher:
    """High-level interface for fetching and caching web pages."""

    def __init__(self, config: 'IfetcherConfig', show_status: bool = True, verbose: bool = False):
        self.cache = URLCache(config)
        self.config = config
        self.status_display = StatusDisplay(show_status=show_status)
        self.verbose = verbose
        self._debug_console = Console(stderr=True) if verbose else None

    async def _fetch_html_url(self, url: str, progress_info=None) -> dict[str, str]:
        """
        Fetch HTML URL using Crawl4AI with DOI extraction.

        Args:
            url: URL to fetch
            progress_info: Optional tuple of (progress_instance, task_id) for progress bars

        Returns:
            Dict containing raw_content, markdown_content, final_url, and doi
        """
        # Full tuple unpack to fail fast on signature changes
        AsyncWebCrawler, CrawlerRunConfig, JsonXPathExtractionStrategy, AsyncLoggerBase, DefaultMarkdownGenerator = _get_crawl4ai_imports()
        domain = url.split('//')[1].split('/')[0] if '//' in url else url

        # First pass uses a lightweight config; heavy config is created only if retrying

        if progress_info is not None:
            progress_instance, task_id = progress_info
            progress_instance.update(task_id, description=f"[green]Fetching HTML from[/green] [bold]{domain}[/bold]")
            granular_logger = _get_quiet_logger()
            # First try: simple browser without JS/wait, headless fast path
            simple_cfg = _build_html_crawler_config(
                timeout_ms=self.config.tools.crawl4ai.timeout * 1000,
                with_js=False,
            )
            browser_simple = _get_browser_config()
            async with AsyncWebCrawler(config=browser_simple, logger=granular_logger) as crawler:
                result = await crawler.arun(url=url, config=simple_cfg)
                if not result.success:
                    raise RuntimeError(f"{result.status_code} error fetching {url}: {result.error_message}")
        else:
            with self.status_display.create_status(f"[dim]Initializing...[/dim]") as status:
                granular_logger = _get_granular_logger(status_display=status)
                # First try: simple browser without JS/wait, headless fast path
                simple_cfg = _build_html_crawler_config(
                    timeout_ms=self.config.tools.crawl4ai.timeout * 1000,
                    with_js=False,
                )
                browser_simple = _get_browser_config()
                async with AsyncWebCrawler(config=browser_simple, logger=granular_logger) as crawler:
                    result = await crawler.arun(url=url, config=simple_cfg)
                    if not result.success:
                        raise RuntimeError(f"{result.status_code} error fetching {url}: {result.error_message}")

        final_url = self._extract_final_url(result, url)
        raw_content = result.html or ''
        raw_markdown_content, cleaned_markdown_content = result.markdown.raw_markdown, result.markdown
        doi = self._extract_doi(result)

        # Decide whether to retry with stealth/full text expansion
        retry, reason = _should_retry_with_stealth(raw_markdown_content, raw_content)
        if self.verbose and self._debug_console:
            self._debug_console.print(
                f"[dim]fetcher[/dim] {domain}: strategy=light len={len(raw_markdown_content or '')} reason={reason}"
            )

        if retry:
            # Build slow/stealth config with JS instrumentation
            slow_cfg = _build_html_crawler_config(
                timeout_ms=self.config.tools.crawl4ai.timeout * 1000,
                with_js=True,
            )
            browser_stealth = _get_browser_config(enable_stealth=True, headless=True)#, headless=False)
            # Update status if available
            if progress_info is not None:
                progress_instance, task_id = progress_info
                progress_instance.update(task_id, description=f"[yellow]Retrying with full text[/yellow] [bold]{domain}[/bold]")
                granular_logger = _get_quiet_logger()
            else:
                granular_logger = _get_granular_logger(status_display=self.status_display)
            async with AsyncWebCrawler(config=browser_stealth, logger=granular_logger) as crawler:
                result2 = await crawler.arun(url=url, config=slow_cfg)
                if result2.success:
                    final_url = self._extract_final_url(result2, url)
                    raw_content = result2.html or raw_content
                    tmp_raw, tmp_cleaned = result2.markdown.raw_markdown, result2.markdown
                    if tmp_raw:
                        raw_markdown_content = tmp_raw
                    if tmp_cleaned:
                        cleaned_markdown_content = tmp_cleaned
                    doi2 = self._extract_doi(result2)
                    if doi2:
                        doi = doi2
                    if self.verbose and self._debug_console:
                        self._debug_console.print(
                            f"[dim]fetcher[/dim] {domain}: strategy=heavy len={len(raw_markdown_content or '')} reason={reason}"
                        )
                else:
                    if self.verbose and self._debug_console:
                        self._debug_console.print(
                            f"[dim]fetcher[/dim] {domain}: heavy_retry_failed status={getattr(result2, 'status_code', '?')}"
                        )

        return {
            'raw_content': raw_content,
            'raw_markdown_content': raw_markdown_content,
            'markdown_content': self._refine_article_content(cleaned_markdown_content),
            'final_url': final_url,
            'doi': doi
        }

    async def _fetch_pdf_url(self, url: str, progress_info=None) -> dict[str, str]:
        """
        Fetch PDF URL using Crawl4AI PDF processing strategy.

        Args:
            url: PDF URL to fetch
            progress_info: Optional tuple of (progress_instance, task_id) for progress bars

        Returns:
            Dict containing raw_content, markdown_content, final_url (no DOI extraction for PDFs yet)
        """
        AsyncWebCrawler, CrawlerRunConfig, PDFCrawlerStrategy, PDFContentScrapingStrategy, _ = _get_crawl4ai_pdf_imports()
        domain = url.split('//')[1].split('/')[0] if '//' in url else url

        pdf_crawler_cfg = PDFCrawlerStrategy()
        pdf_scraping_cfg = PDFContentScrapingStrategy()

        cfg = CrawlerRunConfig(
            scraping_strategy=pdf_scraping_cfg,
            page_timeout=self.config.tools.crawl4ai.timeout * 1000,
            verbose=False
        )

        if progress_info is not None:
            progress_instance, task_id = progress_info
            progress_instance.update(task_id, description=f"[red]Fetching PDF from[/red] [bold]{domain}[/bold]")
            granular_logger = _get_quiet_logger()
            browser_config = _get_browser_config(enable_stealth=True, headless=True)
            async with AsyncWebCrawler(config=browser_config, crawler_strategy=pdf_crawler_cfg, logger=granular_logger) as crawler:
                result = await crawler.arun(url=url, config=cfg)
                if not result.success:
                    raise RuntimeError(f"{result.status_code} error fetching PDF {url}: {result.error_message}")
        else:
            with self.status_display.create_status(f"[dim]Initializing...[/dim]") as status:
                granular_logger = _get_granular_logger(status_display=status)
                browser_config = _get_browser_config(enable_stealth=True, headless=True)
                async with AsyncWebCrawler(config=browser_config, crawler_strategy=pdf_crawler_cfg, logger=granular_logger) as crawler:
                    result = await crawler.arun(url=url, config=cfg)
                    if not result.success:
                        raise RuntimeError(f"{result.status_code} error fetching PDF {url}: {result.error_message}")

        final_url = result.url
        raw_markdown_content, cleaned_markdown_content = result.markdown.raw_markdown, result.markdown
        markdown_content = self._refine_article_content(cleaned_markdown_content)
        raw_content = result.pdf or ''

        return {
            'raw_content': raw_content,
            'raw_markdown_content': raw_markdown_content,
            'markdown_content': markdown_content,
            'content_type': 'application/pdf',
            'final_url': final_url,
            'doi': ''
        }

    def _extract_final_url(self, result, original_url: str) -> str:
        """Extract the final URL after any redirects."""
        final_url = original_url
        if hasattr(result, '_results') and result._results:
            first_result = result._results[0]
            if hasattr(first_result, 'redirected_url') and first_result.redirected_url:
                final_url = first_result.redirected_url
            elif hasattr(first_result, 'url'):
                final_url = first_result.url

        if final_url == original_url and hasattr(result, 'url'):
            final_url = result.url

        return final_url

    def _extract_doi(self, result) -> str:
        """Extract DOI from crawl4ai extracted content."""
        doi = ''
        try:
            if hasattr(result, 'extracted_content') and result.extracted_content:
                extracted_data = json.loads(result.extracted_content)
                for doi_data in extracted_data:

                    for field in ["doi_meta_pub", "doi_meta_cite", "doi_dc_doi", "doi_dc"]:
                        if field in doi_data and doi_data[field]:
                            doi = doi_data[field]
                            break

                    if not doi and "doi_canonical" in doi_data and doi_data["doi_canonical"]:
                        canonical_url = doi_data["doi_canonical"]
                        if 'doi.org/' in canonical_url:
                            doi = canonical_url.split('doi.org/')[1]

                    if doi:
                        break
        except (json.JSONDecodeError, AttributeError, KeyError, IndexError):
            pass
        return doi

    def _refine_article_content(self, markdown: str) -> str:
        """
        Refine article content by removing irrelevant sections using intelligent heading analysis.

        Delegates to the global refine_article_content function.
        """
        return refine_article_content(markdown)

    def _is_pdf_url(self, url: str) -> bool:
        """Check if URL points to a PDF file based on extension."""
        from urllib.parse import urlparse
        return urlparse(url).path.lower().endswith('.pdf')

    def _create_chunks(self, markdown_content: str) -> List[str]:
        """
        Chunk markdown content and return as list of strings.

        Args:
            markdown_content: The markdown content to chunk

        Returns:
            List of chunk strings
        """
        chunker = _get_chunker()

        # Chunk the content
        chunks = chunker(markdown_content)

        # Return just the text content of each chunk
        return [chunk.text for chunk in chunks]

    async def _fetch_and_cache(self, url: str, progress=True, progress_info=None, *, retry: bool = False) -> dict[str, str]:
        """
        Fetch content from URL and cache it based on content type.

        Args:
            url: URL to fetch
            progress: True for status display, False for silent, tuple for progress bar integration
            progress_info: Optional tuple of (progress_instance, task_id) for progress bars

        Returns:
            Dict containing content information
        """
        # Respect previous failure marker unless retrying.
        # Follow any known redirects (from prior successful fetches) before checking.
        if not retry:
            # Check original URL
            if await self.cache.is_failed(url):
                reason = await self.cache.get_failed_reason(url)
                raise PreviousFailure(url, reason or f"Previous failure recorded for URL: {url}")
            # If a redirect mapping exists, check the final URL too
            try:
                visited = set()
                current = url
                for _ in range(3):  # safety cap
                    redir = await self.cache.get_redirect_info(current)
                    if not redir or redir in visited:
                        break
                    visited.add(redir)
                    if await self.cache.is_failed(redir):
                        reason = await self.cache.get_failed_reason(redir)
                        raise PreviousFailure(redir, reason or f"Previous failure recorded for URL: {redir}")
                    current = redir
            except Exception:
                # On any error probing redirect info, fall back to proceeding
                pass

        is_pdf = self._is_pdf_url(url)

        try:
            if progress is True:
                if is_pdf:
                    content_info = await self._fetch_pdf_url(url)
                    content_info['source_type'] = 'pdf'
                else:
                    content_info = await self._fetch_html_url(url)
                    content_info['source_type'] = 'html'
            elif progress is False:
                old_show_status = self.status_display.show_status
                self.status_display.show_status = False
                try:
                    if is_pdf:
                        content_info = await self._fetch_pdf_url(url)
                        content_info['source_type'] = 'pdf'
                    else:
                        content_info = await self._fetch_html_url(url)
                        content_info['source_type'] = 'html'
                finally:
                    self.status_display.show_status = old_show_status
            else:
                if is_pdf:
                    content_info = await self._fetch_pdf_url(url, progress_info=progress)
                    content_info['source_type'] = 'pdf'
                else:
                    content_info = await self._fetch_html_url(url, progress_info=progress)
                    content_info['source_type'] = 'html'
        except Exception as e:
            # Mark failed and re-raise
            try:
                await self.cache.mark_failed(url, reason=str(e))
            finally:
                pass
            raise

        final_url = content_info.get('final_url', url)

        try:
            if is_pdf:
                await self.cache.set_content(url, "pdf", content_info['raw_content'], final_url)
            else:
                await self.cache.set_content(url, "html", content_info['raw_content'], final_url)

            await self.cache.set_content(url, "markdown", content_info['markdown_content'], final_url)

            # Store raw markdown if available (for both HTML and PDF sources)
            if content_info.get('raw_markdown_content'):
                await self.cache.set_content(url, "raw_markdown", content_info['raw_markdown_content'], final_url)

            if content_info.get('doi'):
                await self.cache.set_content(url, "doi", content_info['doi'], final_url)

            # Clear any previous failure marker(s) on success
            await self.cache.clear_failed(url)
            if final_url and final_url != url:
                await self.cache.clear_failed(final_url)
            return content_info
        except Exception as e:
            # Mark as failed and re-raise
            try:
                await self.cache.mark_failed(url, final_url=final_url, reason=str(e))
            finally:
                pass
            raise

    async def get_html(self, url: Union[str, List[str]], progress=True, fail_fast: bool = False, *, retry: bool = False) -> Union[str, List[str]]:
        """
        Get HTML content for single URL or multiple URLs with concurrent fetching.

        Args:
            url: Single URL string or list of URLs to fetch
            progress: True for status display, False for silent, tuple for progress bar integration

        Returns:
            Single content string for single URL, list of content strings for multiple URLs
        """
        if isinstance(url, str):
            if not await self.cache.has_path(url, "html"):
                await self._fetch_and_cache(url, progress=progress, retry=retry)
            return await self.cache.get_content(url, "html")

        elif isinstance(url, list):
            if len(url) == 0:
                return []

            results = await self._fetch_multiple(url, "html", progress=progress, fail_fast=fail_fast, retry=retry)
            if isinstance(progress, tuple):
                # Custom progress returns content directly
                return results

            return self._process_multiple_results(results, progress)

        else:
            raise TypeError(f"url must be str or list[str], got {type(url)}")

    async def get_pdf(self, url: Union[str, List[str]], progress=True, fail_fast: bool = False, *, retry: bool = False) -> Union[str, List[str]]:
        """
        Get PDF content for single URL or multiple URLs with concurrent fetching.

        Args:
            url: Single URL string or list of URLs to fetch
            progress: True for status display, False for silent, tuple for progress bar integration

        Returns:
            Single content string for single URL, list of content strings for multiple URLs
        """
        if isinstance(url, str):
            if not await self.cache.has_path(url, "pdf"):
                await self._fetch_and_cache(url, progress=progress, retry=retry)
            return await self.cache.get_content(url, "pdf")

        elif isinstance(url, list):
            if len(url) == 0:
                return []

            results = await self._fetch_multiple(url, "pdf", progress=progress, fail_fast=fail_fast, retry=retry)
            if isinstance(progress, tuple):
                # Custom progress returns content directly
                return results

            return self._process_multiple_results(results, progress)

        else:
            raise TypeError(f"url must be str or list[str], got {type(url)}")

    async def get_markdown(self, url: Union[str, List[str]], progress=True, fail_fast: bool = False, *, retry: bool = False) -> Union[str, List[str]]:
        """
        Get Markdown content for single URL or multiple URLs with concurrent fetching.
        Automatically detects and handles both HTML and PDF URLs.

        Args:
            url: Single URL string or list of URLs to fetch
            progress: True for status display, False for silent, tuple for progress bar integration

        Returns:
            Single content string for single URL, list of content strings for multiple URLs
        """
        if isinstance(url, str):
            if await self.cache.has_path(url, "markdown"):
                return await self.cache.get_path(url, "markdown")

            # Check if HTML already exists - if so, convert it instead of re-fetching
            if await self.cache.has_path(url, "html"):
                html_content = await self.cache.get_content(url, "html")

                # Create a temporary file with the HTML content
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as temp_file:
                    temp_file.write(html_content)
                    temp_file_path = temp_file.name

                try:
                    # Convert the file:// URL to markdown using existing infrastructure
                    file_url = f"file://{temp_file_path}"
                    content_info = await self._fetch_html_url(file_url, progress_info=progress if isinstance(progress, tuple) else None)

                    # Get redirect info if this URL was originally redirected
                    final_url = await self.cache.get_redirect_info(url)

                    # Store the markdown
                    await self.cache.set_path(url, "markdown", content_info['markdown_content'], final_url)

                    # Also store the raw (pre-processed) markdown sidecar if available
                    if content_info.get('raw_markdown_content'):
                        await self.cache.set_content(url, "raw_markdown", content_info['raw_markdown_content'], final_url)

                    return content_info['markdown_content']
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_file_path)
                    except:
                        pass
            else:
                # No HTML cached, fetch from scratch
                await self._fetch_and_cache(url, progress=progress, retry=retry)
                return await self.cache.get_path(url, "markdown")

        elif isinstance(url, list):
            if len(url) == 0:
                return []

            # Check which URLs need fetching vs are already cached
            urls_to_fetch = []
            results = {}

            for u in url:
                if await self.cache.has_path(u, "markdown"):
                    # Already cached, get from cache
                    try:
                        results[u] = await self.cache.get_path(u, "markdown")
                    except KeyError:
                        urls_to_fetch.append(u)
                else:
                    urls_to_fetch.append(u)

            # Fetch any URLs that aren't cached
            if urls_to_fetch:
                fetch_results = await self._fetch_multiple(urls_to_fetch, "markdown", progress=progress, fail_fast=fail_fast, retry=retry)

                # Map back to original URLs, skipping exceptions
                for i, u in enumerate(urls_to_fetch):
                    if i < len(fetch_results) and not isinstance(fetch_results[i], Exception):
                        results[u] = fetch_results[i]

            # Return results in original order
            return [results[u] for u in url if u in results]

        else:
            raise TypeError(f"url must be str or list[str], got {type(url)}")

    async def get_raw(self, url: str, *, retry: bool = False) -> str:
        """Get raw content (HTML or PDF) for URL, fetching if necessary."""
        # Check cache first
        source_content = await self.cache.get_source_content(url)
        if source_content is not None:
            return source_content

        # Not cached, fetch it
        content_info = await self._fetch_and_cache(url, retry=retry)
        return content_info['raw_content']

    def _process_multiple_results(self, results: List, progress) -> List:
        """
        Filter out exceptions from a mixed results list, returning only content items.

        Args:
            results: List containing a mix of Exception objects and raw content
            progress: Progress mode (unused; kept for signature compatibility)

        Returns:
            List of successful content items (exceptions removed)
        """
        return [result for result in results if not isinstance(result, Exception)]

    async def get_raw_markdown(self, url: str, *, retry: bool = False) -> str:
        """Get raw (pre-cleaned) markdown content for URL, fetching if necessary."""
        # Check cache first
        if await self.cache.has_path(url, "raw_markdown"):
            return await self.cache.get_path(url, "raw_markdown")

        # Not cached, fetch it
        content_info = await self._fetch_and_cache(url, retry=retry)
        return content_info.get('raw_markdown_content', '')

    async def is_cached(self, url: str) -> bool:
        """Check if URL is cached (any content type)."""
        return await self.cache.has_url(url)

    async def get_source_type(self, url: str) -> Optional[str]:
        """Get the source content type for a cached URL."""
        return await self.cache.get_source_type(url)

    async def clear_cache(self, url: str) -> None:
        """Clear cached content for URL."""
        await self.cache.clear_url(url)

    async def get_chunks(self, url: Union[str, List[str]], progress=True, fail_fast: bool = False, *, retry: bool = False) -> Union[List[str], List[List[str]]]:
        """
        Get chunked content for single URL or multiple URLs as lists of strings.
        Automatically fetches markdown first if not cached, then chunks it.

        Args:
            url: Single URL string or list of URLs to fetch
            progress: True for status display, False for silent, tuple for progress bar integration

        Returns:
            Single list of strings for single URL, list of lists of strings for multiple URLs
        """
        if isinstance(url, str):
            # Check if chunks are already cached
            if await self.cache.has_path(url, "chunks"):
                return await self.cache.get_content(url, "chunks")

            # Get markdown content first
            markdown_content = await self.get_markdown(url, progress=progress, retry=retry)

            # Get final URL from redirect info if it exists
            final_url = await self.cache.get_redirect_info(url)

            # Chunk it and cache (silently)
            chunks = self._create_chunks(markdown_content)
            await self.cache.set_content(url, "chunks", chunks, final_url)

            return chunks

        elif isinstance(url, list):
            if len(url) == 0:
                return []

            results = await self._fetch_multiple(url, "chunks", progress=progress, fail_fast=fail_fast, retry=retry)
            return self._process_multiple_results(results, progress)

        else:
            raise TypeError(f"url must be str or list[str], got {type(url)}")

    async def prefetch(self, urls: list[str], progress=True) -> None:
        """
        Prefetch multiple URLs for faster subsequent access.

        Args:
            urls: List of URLs to prefetch
            progress: True for status display, False for silent mode
        """
        for url in urls:
            try:
                await self.get_markdown(url, progress=progress)
            except Exception as e:
                print(f"Failed to prefetch {url}: {e}")

    async def get_doi(self, url: str, *, retry: bool = False) -> Optional[str]:
        """
        Get DOI for URL if available.

        Args:
            url: URL to get DOI for

        Returns:
            DOI string if found, None otherwise
        """
        if not await self.cache.has_path(url, "doi") and not await self.cache.has_url(url):
            await self._fetch_and_cache(url, retry=retry)
        return await self.cache.get_content(url, "doi")

    async def _get_content_by_type(self, url: str, content_type: str, progress_info=None, *, retry: bool = False) -> Any:
        """Get content of specified type, fetching if necessary."""
        if content_type == "html":
            if not await self.cache.has_path(url, "html"):
                await self._fetch_and_cache(url, progress=progress_info, retry=retry)
            return await self.cache.get_path(url, "html")
        elif content_type == "pdf":
            if not await self.cache.has_path(url, "pdf"):
                await self._fetch_and_cache(url, progress=progress_info, retry=retry)
            return await self.cache.get_path(url, "pdf")
        elif content_type == "markdown":
            if await self.cache.has_path(url, "markdown"):
                return await self.cache.get_path(url, "markdown")
            else:
                # Fetch appropriate content type to generate markdown
                if self._is_pdf_url(url):
                    if not await self.cache.has_path(url, "pdf"):
                        await self._fetch_and_cache(url, progress=progress_info, retry=retry)
                else:
                    if not await self.cache.has_path(url, "html"):
                        await self._fetch_and_cache(url, progress=progress_info, retry=retry)
                return await self.cache.get_path(url, "markdown")
        elif content_type == "chunks":
            if await self.cache.has_path(url, "chunks"):
                return await self.cache.get_content(url, "chunks")
            else:
                # Get markdown content first
                if await self.cache.has_path(url, "markdown"):
                    markdown_content = await self.cache.get_path(url, "markdown")
                else:
                    # Fetch appropriate content type to generate markdown
                    if self._is_pdf_url(url):
                        if not await self.cache.has_path(url, "pdf"):
                            await self._fetch_and_cache(url, progress=progress_info, retry=retry)
                    else:
                        if not await self.cache.has_path(url, "html"):
                            await self._fetch_and_cache(url, progress=progress_info, retry=retry)
                    markdown_content = await self.cache.get_path(url, "markdown")

                # Chunk the markdown content
                chunks = self._create_chunks(markdown_content)
                await self.cache.set_content(url, "chunks", chunks)
                return chunks
        else:
            return await self.get_raw(url, retry=retry)

    def _calculate_content_size(self, content: Any, content_type: str) -> int:
        """Calculate appropriate size for different content types."""
        if content_type == "chunks":
            return sum(len(chunk) for chunk in content)
        return len(content)

    async def _fetch_multiple(self, urls: List[str], content_type: str, progress=None, max_concurrent: int = 5, fail_fast: bool = False, *, retry: bool = False) -> List:
        """
        Unified method to fetch multiple URLs with different progress modes.

        Args:
            urls: List of URLs to fetch
            content_type: Type of content ("html", "pdf", "markdown", "chunks")
            progress: True for default progress, False for silent, tuple for custom progress
            max_concurrent: Maximum concurrent fetches

        Returns:
            List of raw content items or Exception objects (uniform across modes)
        """
        if fail_fast:
            # Sequential processing with early abort on first failure
            results: list = []
            for url in urls:
                try:
                    content = await self._get_content_by_type(url, content_type, progress_info=False, retry=retry)
                    # Always append raw content in fail-fast mode
                    results.append(content)
                except Exception as e:
                    # Return immediately with the exception to signal fail-fast
                    results.append(e)
                    return results
            return results
        if progress is True:
            # Default progress with concurrent processing
            results = await fetch_urls_concurrent_with_progress(
                urls,
                self.config,
                content_type=content_type,
                max_concurrent=max_concurrent,
                retry=retry,
                verbose=self.verbose
            )
            # Already raw content or exceptions
            return results
        elif progress is False:
            # Silent concurrent processing
            import asyncio
            semaphore = asyncio.Semaphore(max_concurrent)

            async def fetch_one(url: str):
                async with semaphore:
                    try:
                        content = await self._get_content_by_type(url, content_type, progress_info=False, retry=retry)
                        return content
                    except Exception as e:
                        return e

            tasks = [fetch_one(url) for url in urls]
            results = await asyncio.gather(*tasks)
            return results
        else:
            # Custom progress (sequential processing)
            progress_instance, task_id = progress
            results = []

            for i, url in enumerate(urls):
                try:
                    domain = url.split('//')[1].split('/')[0] if '//' in url else url
                    progress_instance.update(task_id, description=f"[blue]Fetching {content_type} from[/blue] [bold]{domain}[/bold] ({i+1}/{len(urls)})")

                    # Handle content fetch without special chunking messages
                    content = await self._get_content_by_type(url, content_type, progress, retry=retry)

                    results.append(content)
                except Exception as e:
                    # Return exception directly instead of hiding it
                    results.append(e)

            return results
async def fetch_urls_with_progress(
    urls: list[str],
    config: 'IfetcherConfig',
    content_type: str = "html",
    progress_description: str = "Fetching URLs...",
    verbose: bool = False,
    retry: bool = False,
) -> list:
    """
    Fetch multiple URLs with clean progress display.

    Args:
        urls: List of URLs to fetch
        config: Configuration for PageFetcher
        content_type: Type of content to fetch ("html", "pdf", "markdown")
        progress_description: Description for progress bar

    Returns:
        List of raw content items or Exception objects (used for UI-only progress)
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
    import math
    from urllib.parse import urlparse

    # Use direct method calls instead of string lookup
    fetcher = PageFetcher(config)

    # Calculate optimal domain width using 75th percentile for better preservation
    domains = [urlparse(url).netloc for url in urls]
    domain_lengths = [len(domain) for domain in domains if domain]
    if domain_lengths:
        import statistics
        percentile_75 = statistics.quantiles(domain_lengths, n=4)[2] if len(domain_lengths) > 1 else domain_lengths[0]
        optimal_width = min(20, int(percentile_75))
    else:
        optimal_width = 15

    def format_domain(url: str) -> str:
        domain = urlparse(url).netloc
        if len(domain) <= optimal_width:
            return domain.ljust(optimal_width)
        else:
            return domain[:optimal_width-1] + "…"

    results: list = []

    _prog_console = Console(stderr=True)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="blue", finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=False,
        refresh_per_second=8,
        console=_prog_console,
    ) as progress:

        # Start with a neutral initializing state, then switch to fetching
        task = progress.add_task("[dim]Initializing...[/dim]", total=len(urls))
        fetcher = PageFetcher(config, show_status=False, verbose=verbose)

        for url in urls:
            try:
                # Switch to fetching status per-URL
                formatted_domain = format_domain(url)
                progress.update(task, description=f"[blue]Fetching...[/blue] {formatted_domain}")
                if content_type == "html":
                    content = await fetcher.get_html(url, progress=(progress, task), retry=retry)
                elif content_type == "pdf":
                    content = await fetcher.get_pdf(url, progress=(progress, task), retry=retry)
                elif content_type == "markdown":
                    content = await fetcher.get_markdown(url, progress=(progress, task), retry=retry)
                elif content_type == "chunks":
                    content = await fetcher.get_chunks(url, progress=(progress, task), retry=retry)
                else:
                    content = await fetcher.get_raw(url, retry=retry)

                # Calculate size appropriately based on content type
                if content_type == "chunks":
                    size = sum(len(chunk) for chunk in content)
                else:
                    size = len(content)

                results.append(content)

                formatted_domain = format_domain(url)
                progress.update(task, description=f"[green]✓ {formatted_domain} ({len(content):,} chars)")

            except Exception as e:
                results.append(e)
                domain = url.split('//')[1].split('/')[0] if '//' in url else url
                progress.update(task, description=f"[red]✗ {domain} (failed)")

            progress.advance(task)

    return results

async def fetch_urls_concurrent_with_progress(
    urls: list[str],
    config: 'IfetcherConfig',
    content_type: str = "html",
    max_concurrent: int = 5,
    verbose: bool = False,
    retry: bool = False,
) -> list:
    """
    Fetch multiple URLs concurrently with progress display.

    Args:
        urls: List of URLs to fetch
        config: Configuration for PageFetcher
        content_type: Type of content to fetch
        max_concurrent: Maximum concurrent fetches

    Returns:
        List of raw content items or Exception objects (used for UI-only progress)
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
    import asyncio
    import math
    from urllib.parse import urlparse

    # Use direct method calls instead of string lookup

    # Calculate optimal domain width using 75th percentile for better preservation
    domains = [urlparse(url).netloc for url in urls]
    domain_lengths = [len(domain) for domain in domains if domain]
    if domain_lengths:
        import statistics
        percentile_75 = statistics.quantiles(domain_lengths, n=4)[2] if len(domain_lengths) > 1 else domain_lengths[0]
        optimal_width = min(20, int(percentile_75))
    else:
        optimal_width = 15

    def format_domain(url: str) -> str:
        domain = urlparse(url).netloc
        if len(domain) <= optimal_width:
            return domain.ljust(optimal_width)
        else:
            return domain[:optimal_width-1] + "…"

    _prog_console = Console(stderr=True)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="blue", finished_style="green"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        transient=False,
        refresh_per_second=8,
        console=_prog_console,
    ) as progress:

        # Start with a neutral initializing state, then switch to fetching
        task = progress.add_task("[dim]Initializing...[/dim]", total=len(urls))
        completed = {"count": 0}

        async def fetch_one(url: str):
            # Use progress bar; avoid per-URL spinners to reduce noise
            fetcher = PageFetcher(config, show_status=False, verbose=verbose)
            formatted_domain = format_domain(url)

            try:
                if content_type == "html":
                    content = await fetcher.get_html(url, progress=False, retry=retry)
                elif content_type == "pdf":
                    content = await fetcher.get_pdf(url, progress=False, retry=retry)
                elif content_type == "markdown":
                    content = await fetcher.get_markdown(url, progress=False, retry=retry)
                elif content_type == "chunks":
                    content = await fetcher.get_chunks(url, progress=False, retry=retry)
                else:
                    content = await fetcher.get_raw(url, retry=retry)

                completed["count"] += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"[blue]Fetching... ({completed['count']}/{len(urls)}) {formatted_domain}"
                )

                # Size calculated only for progress description above; return raw content
                return content

            except Exception as e:
                completed["count"] += 1
                progress.update(
                    task,
                    advance=1,
                    description=f"[blue]Fetching... ({completed['count']}/{len(urls)}) - Failed: {formatted_domain}"
                )

                return e

        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_limit(url):
            async with semaphore:
                return await fetch_one(url)

        # Switch to fetching state as tasks are scheduled
        progress.update(task, description=f"[blue]Fetching...[/blue] (0/{len(urls)})")
        tasks = [fetch_with_limit(url) for url in urls]
        results = await asyncio.gather(*tasks)

    return results

    async def list_cached_urls(self) -> list[str]:
        """Get a list of all cached URLs."""
        return await self.cache.list_cached_urls()

    async def get_doi(self, url: str) -> Optional[str]:
        """Get DOI for URL if available."""
        if not await self.cache.has_path(url, "doi") and not await self.cache.has_url(url):
            # Try to fetch and extract DOI
            await self._fetch_and_cache(url)

        return await self.cache.get_content(url, "doi")
