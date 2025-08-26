"""Content processing and refinement for academic articles."""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# Content processing constants
RELEVANT_HEADINGS = {
    "abstract",
    "summary",
    "introduction",
    "methods",
    "methodology",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "background",
    "objectives",
    "findings",
    "analysis",
    "materials",
    "procedure",
    "approach",
    "appendix",
    "appendices",
    "limitations",
    "future work",
    "implications",
    "main text",
}

IRRELEVANT_HEADINGS = {
    "references",
    "keywords",
    "bibliography",
    "citations",
    "authors",
    "author",
    "acknowledgements",
    "acknowledgments",
    "share",
    "sharing",
    "content link",
    "funding",
    "conflicts",
    "conflict of interest",
    "competing interests",
    "data availability",
    "supplementary",
    "supporting information",
    "copyright",
    "license",
    "permissions",
    "ethics",
    "rights",
    "affiliations",
    "corresponding",
    "license",
    "licence",
    "cite",
    "how to cite",
    "cite this",
    "cited by",
    "reprints",
    "related articles",
    "similar articles",
    "publication history",
    "publication types",
    "notes",
    "comments",
}

IRRELEVANT_PATTERNS = [
    # Site chrome and navigation (exclude paywall/login which live in PAYWALL_PATTERNS)
    r".*cookies?.*",
    r".*privacy.*",
    r".*terms.*",
    r".*subscribe.*",
    r".*newsletter.*",
    r".*follow.*",
    r".*social.*",
    r".*menu.*",
    r".*navigation.*",
    r".*search.*",
    r".*contact.*",
    r".*about.*",
    r".*read.*content.*",
    r"similar content.*",
    r".*viewed by others",
    r"recommended.*",
    r"supplementary\s+.*",
    r"supplemental\s+.*",
    r".*metrics.*",
    r".*altmetric.*",
    r"declaration\s+.*",
    r".*availability.*",
    r".*privacy.*",
    r".*password.*",
    r".*username.*",
    r"comment\s?.*",
]

PAYWALL_PATTERNS = [
    r".*log ?in.*",
    r".*sign\s*in.*",
    r".*sign\s*up.*",
    r".*register.*",
    r".*create.*account.*",
    r".*free.*account.*",
    r".*subscription.*",
    r".*paywall.*",
    r".*access.*denied.*",
    r".*get\s+.*access.*",
    r".*premium.*content.*",
    r".*unlock.*content.*",
    r".*full.*access.*",
    r".*purchase.*",
]

INLINE_SHORT_LINE_MAXLEN = 140
CHROME_INLINE_KEYWORDS = [
    "sign in",
    "sign up",
    "log in",
    "log-in",
    "register",
    "subscribe",
    "forgot password",
    "username",
    "password",
    "privacy",
    "terms",
    "cookies",
    "share on",
    "facebook",
    "twitter",
    "linkedin",
    "email",
]

OVER_TRIM_THRESHOLD = 0.3


@dataclass
class HeadingInfo:
    """Information about a heading in markdown content."""

    text: str
    level: int
    start_pos: int
    end_pos: int
    full_match: str
    is_relevant: bool = False
    is_irrelevant: bool = False
    is_paywall: bool = False


class ContentProcessor:
    """Processes and refines academic content by removing site chrome and focusing on core article content."""

    def refine_article(self, markdown: str) -> str:
        """Main entry point - orchestrates the refinement process."""
        headings = self._extract_and_classify_headings(markdown)
        if not headings:
            return markdown

        # Try duplicate content removal first (safe operation)
        try:
            markdown = self._remove_duplicate_intro_sections(markdown, headings)
            headings = self._extract_and_classify_headings(
                markdown
            )  # Re-parse after trim
        except Exception:
            pass  # Fall back to original content

        return self._refine_with_headings(markdown, headings)

    def _extract_and_classify_headings(self, markdown: str) -> List[HeadingInfo]:
        """Extract headings and classify them in one pass."""
        headings = []
        heading_pattern = r"^(#{1,6})\s+(.+?)(?:\s*\{[^}]*\})?\s*$"

        for match in re.finditer(heading_pattern, markdown, re.MULTILINE):
            heading_info = HeadingInfo(
                text=match.group(2).strip(),
                level=len(match.group(1)),
                start_pos=match.start(),
                end_pos=match.end(),
                full_match=match.group(0),
            )
            self._classify_heading(heading_info)
            headings.append(heading_info)
        return headings

    def _classify_heading(self, heading_info: HeadingInfo) -> None:
        """Classify a heading as relevant, irrelevant, or paywall content."""
        text_lower = heading_info.text.lower()
        all_irrelevant_patterns = IRRELEVANT_PATTERNS + PAYWALL_PATTERNS

        # Check for irrelevant patterns first (higher priority)
        heading_info.is_irrelevant = any(
            irrel in text_lower for irrel in IRRELEVANT_HEADINGS
        ) or any(
            re.match(pattern, text_lower, re.IGNORECASE)
            for pattern in all_irrelevant_patterns
        )

        # Check for paywall patterns
        heading_info.is_paywall = any(
            re.match(pattern, text_lower, re.IGNORECASE) for pattern in PAYWALL_PATTERNS
        )

        # Only check for relevant patterns if not already marked as irrelevant
        heading_info.is_relevant = not heading_info.is_irrelevant and any(
            rel in text_lower for rel in RELEVANT_HEADINGS
        )

    def _remove_duplicate_intro_sections(
        self, markdown: str, headings: List[HeadingInfo]
    ) -> str:
        """Remove duplicate introductory sections using several heuristics."""
        relevant_headings = [h for h in headings if h.is_relevant]
        if len(relevant_headings) < 2:
            return markdown

        # Section-repeat heuristic
        first_relevant = relevant_headings[0]
        second_relevant = relevant_headings[1]

        if first_relevant.text.lower() == second_relevant.text.lower():
            preceding_larger = self._find_preceding_larger_heading(
                headings, second_relevant
            )
            if (
                preceding_larger
                and first_relevant.start_pos < preceding_larger.start_pos
            ):
                return markdown[preceding_larger.start_pos :]

        # Fallback heuristics
        h1_headings = [h for h in headings if h.level == 1]
        if len(h1_headings) >= 2:
            first_h1, second_h1 = h1_headings[0], h1_headings[1]
            # Check if Abstract appears before second title
            abstract_headings = [
                h for h in relevant_headings if "abstract" in h.text.lower()
            ]
            if (
                abstract_headings
                and abstract_headings[0].start_pos < second_h1.start_pos
            ):
                return markdown[second_h1.start_pos :]

        # Two Abstract headings fallback
        abstract_headings = [
            h for h in relevant_headings if "abstract" in h.text.lower()
        ]
        if len(abstract_headings) >= 2:
            return markdown[abstract_headings[1].start_pos :]

        return markdown

    def _find_preceding_larger_heading(
        self, headings: List[HeadingInfo], target_heading: HeadingInfo
    ) -> Optional[HeadingInfo]:
        """Find the nearest heading with a lower level number (larger visual rank) before the target."""
        candidates = [
            h
            for h in headings
            if h.start_pos < target_heading.start_pos and h.level < target_heading.level
        ]
        return max(candidates, key=lambda h: h.start_pos) if candidates else None

    def _refine_with_headings(self, markdown: str, headings: List[HeadingInfo]) -> str:
        """Core refinement logic with well-defined phases."""
        relevant_headings = [h for h in headings if h.is_relevant]

        if not relevant_headings:
            return self._handle_no_relevant_sections(markdown, headings)

        core_start, core_end = self._find_core_boundaries(
            markdown, headings, relevant_headings
        )
        refined = self._remove_sections_by_regions(
            markdown, headings, core_start, core_end
        )

        # Over-trim protection
        if len(refined) < len(markdown) * OVER_TRIM_THRESHOLD:
            if not self._has_academic_structure(headings):
                return markdown  # Fallback for non-academic content

        return self._cleanup_inline_chrome(refined)

    def _handle_no_relevant_sections(
        self, markdown: str, headings: List[HeadingInfo]
    ) -> str:
        """Handle content with no relevant academic sections."""
        paywall_headings = [h for h in headings if h.is_paywall]
        if paywall_headings:
            return ""  # Return empty string for paywall content

        # Remove irrelevant sections but keep everything else
        return self._remove_irrelevant_sections_aggressively(markdown, headings)

    def _find_core_boundaries(
        self, markdown: str, headings: List[HeadingInfo], relevant: List[HeadingInfo]
    ) -> Tuple[int, int]:
        """Find start/end positions of core article content."""
        # Abstract/Summary/Introduction preference with title preservation
        preferred_starters = ["abstract", "summary", "introduction"]
        core_heading = next(
            (
                h
                for h in relevant
                if any(starter in h.text.lower() for starter in preferred_starters)
            ),
            relevant[0] if relevant else None,
        )

        if not core_heading:
            return 0, len(markdown)

        core_start = self._find_start_with_title_preservation(headings, core_heading)
        core_end = self._find_core_end(headings, relevant[-1], len(markdown))
        return core_start, core_end

    def _find_start_with_title_preservation(
        self, headings: List[HeadingInfo], core_heading: HeadingInfo
    ) -> int:
        """Find core start position, preserving title heading if present."""
        # Look for a title heading immediately preceding the core heading
        title_heading = self._find_preceding_larger_heading(headings, core_heading)
        if title_heading:
            # Preserve title heading, remove content between title and core
            return title_heading.start_pos
        return core_heading.start_pos

    def _find_core_end(
        self, headings: List[HeadingInfo], last_relevant: HeadingInfo, markdown_len: int
    ) -> int:
        """Find the end position of core content."""
        # Find first irrelevant heading after the last relevant section
        for heading in headings:
            if heading.start_pos > last_relevant.start_pos and heading.is_irrelevant:
                return heading.start_pos
        return markdown_len

    def _remove_sections_by_regions(
        self, markdown: str, headings: List[HeadingInfo], core_start: int, core_end: int
    ) -> str:
        """Remove sections by regions: before core, within core, and after core."""
        # Start with core region
        result = markdown[core_start:core_end]

        # Remove irrelevant sections within the core
        result = self._remove_irrelevant_sections_within_core(
            result, headings, core_start
        )

        # Remove reference-like sections if academic structure detected
        if self._has_academic_structure(headings):
            result = self._remove_reference_sections(result)

        # Handle sections after core_end (preserve relevant sections like Appendix)
        after_core = markdown[core_end:]
        if after_core:
            cleaned_after_core = self._remove_irrelevant_sections_selectively(
                after_core, headings, core_end
            )
            result += cleaned_after_core

        return result

    def _remove_irrelevant_sections_within_core(
        self, markdown: str, all_headings: List[HeadingInfo], offset: int
    ) -> str:
        """Remove irrelevant sections within the core region."""
        # Re-extract headings within this markdown segment
        core_headings = self._extract_and_classify_headings(markdown)

        # Remove each irrelevant section
        for heading in reversed(core_headings):  # Reverse to maintain positions
            if heading.is_irrelevant:
                next_heading_pos = self._find_next_same_or_higher_level_heading(
                    core_headings, heading
                )
                if next_heading_pos is not None:
                    # Remove from heading to next same-or-higher level
                    markdown = (
                        markdown[: heading.start_pos] + markdown[next_heading_pos:]
                    )
                else:
                    # Remove to end of content
                    markdown = markdown[: heading.start_pos]

        return markdown

    def _find_next_same_or_higher_level_heading(
        self, headings: List[HeadingInfo], current: HeadingInfo
    ) -> Optional[int]:
        """Find the next heading at same or higher level (lower level number)."""
        for heading in headings:
            if heading.start_pos > current.start_pos and heading.level <= current.level:
                return heading.start_pos
        return None

    def _remove_reference_sections(self, markdown: str) -> str:
        """Remove reference-like sections anywhere in the content."""
        headings = self._extract_and_classify_headings(markdown)
        reference_patterns = ["references", "bibliography", "citations"]

        for heading in reversed(headings):  # Reverse to maintain positions
            if any(pattern in heading.text.lower() for pattern in reference_patterns):
                next_pos = self._find_next_same_or_higher_level_heading(
                    headings, heading
                )
                if next_pos is not None:
                    markdown = markdown[: heading.start_pos] + markdown[next_pos:]
                else:
                    markdown = markdown[: heading.start_pos]

        return markdown

    def _remove_irrelevant_sections_selectively(
        self, markdown: str, all_headings: List[HeadingInfo], offset: int
    ) -> str:
        """Remove only irrelevant sections, preserving relevant ones like Appendix."""
        headings = self._extract_and_classify_headings(markdown)

        for heading in reversed(headings):
            if heading.is_irrelevant:
                next_pos = self._find_next_same_or_higher_level_heading(
                    headings, heading
                )
                if next_pos is not None:
                    markdown = markdown[: heading.start_pos] + markdown[next_pos:]
                else:
                    markdown = markdown[: heading.start_pos]

        return markdown

    def _remove_irrelevant_sections_aggressively(
        self, markdown: str, headings: List[HeadingInfo]
    ) -> str:
        """Aggressively remove irrelevant sections when no relevant sections found."""
        for heading in reversed(headings):
            if heading.is_irrelevant:
                next_pos = self._find_next_same_or_higher_level_heading(
                    headings, heading
                )
                if next_pos is not None:
                    markdown = markdown[: heading.start_pos] + markdown[next_pos:]
                else:
                    markdown = markdown[: heading.start_pos]
        return markdown

    def _has_academic_structure(self, headings: List[HeadingInfo]) -> bool:
        """Check if the content has clear academic structure."""
        relevant_count = sum(1 for h in headings if h.is_relevant)
        academic_terms = [
            "abstract",
            "introduction",
            "methods",
            "results",
            "discussion",
            "conclusion",
        ]
        academic_headings = sum(
            1
            for h in headings
            if h.is_relevant and any(term in h.text.lower() for term in academic_terms)
        )
        return relevant_count >= 3 or academic_headings >= 2

    def _cleanup_inline_chrome(self, text: str) -> str:
        """Remove inline chrome elements like login prompts and social media links."""
        lines = text.splitlines()
        cleaned = []
        skip_until_next_heading = False
        heading_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

        for line in lines:
            match = heading_re.match(line)
            if match:
                skip_until_next_heading = False
                heading_text = match.group(2).strip().lower()
                # If heading itself is irrelevant, skip this heading and section content
                if (
                    any(h in heading_text for h in IRRELEVANT_HEADINGS)
                    or any(
                        re.match(p, heading_text, re.IGNORECASE)
                        for p in IRRELEVANT_PATTERNS
                    )
                    or any(
                        re.match(p, heading_text, re.IGNORECASE)
                        for p in PAYWALL_PATTERNS
                    )
                ):
                    skip_until_next_heading = True
                    continue
                cleaned.append(line)
                continue

            if skip_until_next_heading:
                continue

            stripped = line.strip()
            # Drop obvious short chrome lines
            if stripped and len(stripped) <= INLINE_SHORT_LINE_MAXLEN:
                lower_text = stripped.lower()
                if any(keyword in lower_text for keyword in CHROME_INLINE_KEYWORDS):
                    continue
            cleaned.append(line)

        return "\n".join(cleaned).strip()


def extract_headings(markdown: str) -> List[Dict]:
    """
    Extract all headings from markdown content with metadata.

    Legacy function for backward compatibility.
    """
    processor = ContentProcessor()
    headings = processor._extract_and_classify_headings(markdown)

    return [
        {
            "text": h.text,
            "level": h.level,
            "start_pos": h.start_pos,
            "end_pos": h.end_pos,
            "full_match": h.full_match,
            "is_relevant": h.is_relevant,
            "is_irrelevant": h.is_irrelevant,
            "is_paywall": h.is_paywall,
        }
        for h in headings
    ]


def classify_heading_relevance(heading_text: str) -> Dict:
    """
    Classify a heading as relevant, irrelevant, or paywall content.

    Legacy function for backward compatibility.
    """
    processor = ContentProcessor()
    heading_info = HeadingInfo(
        text=heading_text,
        level=1,  # Level doesn't matter for classification
        start_pos=0,
        end_pos=0,
        full_match="",
    )
    processor._classify_heading(heading_info)

    return {
        "is_relevant": heading_info.is_relevant,
        "is_irrelevant": heading_info.is_irrelevant,
        "is_paywall": heading_info.is_paywall,
    }


def refine_article_content(markdown: str) -> str:
    """
    Legacy function for backward compatibility.
    Trim a page's Markdown to the core scholarly article while removing site chrome.
    """
    processor = ContentProcessor()
    return processor.refine_article(markdown)
