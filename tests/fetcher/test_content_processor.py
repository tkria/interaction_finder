"""
Tests for academic content processing functionality.

Tests cover:
- HeadingInfo dataclass functionality
- Heading extraction and classification
- Content refinement and article processing
- Chrome removal and cleanup
- Academic structure detection
- Paywall content handling
- Legacy function compatibility
"""

import pytest
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from interaction_finder.fetcher.content_processor import (
    ContentProcessor,
    HeadingInfo,
    extract_headings,
    classify_heading_relevance,
    refine_article_content,
    RELEVANT_HEADINGS,
    IRRELEVANT_HEADINGS,
    PAYWALL_PATTERNS,
    IRRELEVANT_PATTERNS,
    CHROME_INLINE_KEYWORDS,
    OVER_TRIM_THRESHOLD,
    INLINE_SHORT_LINE_MAXLEN,
)


class TestHeadingInfo:
    """Test HeadingInfo dataclass functionality."""

    def test_heading_info_creation_basic(self):
        """Test basic HeadingInfo creation."""
        heading = HeadingInfo(
            text="Introduction",
            level=1,
            start_pos=0,
            end_pos=20,
            full_match="# Introduction",
        )

        assert heading.text == "Introduction"
        assert heading.level == 1
        assert heading.start_pos == 0
        assert heading.end_pos == 20
        assert heading.full_match == "# Introduction"
        assert heading.is_relevant is False  # Default
        assert heading.is_irrelevant is False  # Default
        assert heading.is_paywall is False  # Default

    def test_heading_info_creation_with_flags(self):
        """Test HeadingInfo creation with classification flags."""
        heading = HeadingInfo(
            text="Methods",
            level=2,
            start_pos=100,
            end_pos=120,
            full_match="## Methods",
            is_relevant=True,
            is_irrelevant=False,
            is_paywall=False,
        )

        assert heading.text == "Methods"
        assert heading.is_relevant is True
        assert heading.is_irrelevant is False
        assert heading.is_paywall is False


class TestHeadingExtraction:
    """Test heading extraction functionality."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_extract_basic_headings(self, processor):
        """Test extraction of basic markdown headings."""
        markdown = """# Main Title
        
## Introduction

Some content here.

### Methods

More content.

#### Results

Final content.
"""
        headings = processor._extract_and_classify_headings(markdown)

        assert len(headings) == 4
        assert headings[0].text == "Main Title"
        assert headings[0].level == 1
        assert headings[1].text == "Introduction"
        assert headings[1].level == 2
        assert headings[2].text == "Methods"
        assert headings[2].level == 3
        assert headings[3].text == "Results"
        assert headings[3].level == 4

    def test_extract_headings_with_attributes(self, processor):
        """Test extraction of headings with markdown attributes."""
        markdown = """# Title {#main-title}

## Section {.class-name data-id="123"}

### Subsection {style="color: blue"}
"""
        headings = processor._extract_and_classify_headings(markdown)

        assert len(headings) == 3
        assert headings[0].text == "Title"
        assert headings[1].text == "Section"
        assert headings[2].text == "Subsection"

    def test_extract_no_headings(self, processor):
        """Test extraction from content with no headings."""
        markdown = "This is just regular text with no headings at all."

        headings = processor._extract_and_classify_headings(markdown)

        assert len(headings) == 0

    def test_extract_empty_content(self, processor):
        """Test extraction from empty content."""
        headings = processor._extract_and_classify_headings("")

        assert len(headings) == 0


class TestHeadingClassification:
    """Test heading classification functionality."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_classify_relevant_headings(self, processor):
        """Test classification of relevant academic headings."""
        relevant_texts = [
            "Abstract",
            "Introduction",
            "Methods",
            "Results",
            "Discussion",
            "Conclusions",
        ]

        for text in relevant_texts:
            heading = HeadingInfo(
                text=text, level=2, start_pos=0, end_pos=10, full_match=""
            )
            processor._classify_heading(heading)
            assert heading.is_relevant is True
            assert heading.is_irrelevant is False

    def test_classify_irrelevant_headings(self, processor):
        """Test classification of irrelevant headings."""
        irrelevant_texts = [
            "References",
            "Acknowledgments",
            "Authors",
            "Keywords",
            "Bibliography",
        ]

        for text in irrelevant_texts:
            heading = HeadingInfo(
                text=text, level=2, start_pos=0, end_pos=10, full_match=""
            )
            processor._classify_heading(heading)
            assert heading.is_irrelevant is True
            assert heading.is_relevant is False

    def test_classify_paywall_headings(self, processor):
        """Test classification of paywall-related headings."""
        paywall_texts = [
            "Log In",
            "Sign Up",
            "Register",
            "Subscription Required",
            "Get Access",
        ]

        for text in paywall_texts:
            heading = HeadingInfo(
                text=text, level=2, start_pos=0, end_pos=10, full_match=""
            )
            processor._classify_heading(heading)
            assert heading.is_paywall is True
            assert heading.is_irrelevant is True  # Paywall headings are also irrelevant

    def test_classify_case_insensitive(self, processor):
        """Test that classification is case insensitive."""
        heading = HeadingInfo(
            text="ABSTRACT", level=1, start_pos=0, end_pos=10, full_match=""
        )
        processor._classify_heading(heading)
        assert heading.is_relevant is True

        heading = HeadingInfo(
            text="references", level=1, start_pos=0, end_pos=10, full_match=""
        )
        processor._classify_heading(heading)
        assert heading.is_irrelevant is True

    def test_classify_neutral_headings(self, processor):
        """Test classification of neutral headings that don't match any pattern."""
        heading = HeadingInfo(
            text="Custom Section", level=2, start_pos=0, end_pos=10, full_match=""
        )
        processor._classify_heading(heading)

        assert heading.is_relevant is False
        assert heading.is_irrelevant is False
        assert heading.is_paywall is False


class TestContentRefinement:
    """Test content refinement and article processing."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_refine_simple_academic_content(self, processor):
        """Test refinement of simple academic content."""
        markdown = """# Research Paper Title

## Abstract

This is the abstract of the paper.

## Introduction

This is the introduction section.

## Methods

This is the methods section.

## References

1. Reference 1
2. Reference 2
"""
        result = processor.refine_article(markdown)

        # Should preserve core academic content including title with 'research'
        assert "Research Paper Title" in result
        assert "Abstract" in result
        assert "Introduction" in result
        assert "Methods" in result
        # Should remove references section
        assert "Reference 1" not in result

    def test_refine_content_with_chrome(self, processor):
        """Test refinement of content with site chrome."""
        markdown = """# Site Header

## Sign In

Please log in to access content.

## Abstract

This is the abstract.

## Introduction 

Main content here.

## Subscribe to Newsletter

Get updates via email.

## Acknowledgments

Thanks to contributors.
"""
        result = processor.refine_article(markdown)

        # Should preserve academic content
        assert "Abstract" in result
        assert "Introduction" in result
        # Should remove chrome and acknowledgments
        assert "Sign In" not in result
        assert "Subscribe" not in result
        assert "Acknowledgments" not in result

    def test_refine_paywall_content(self, processor):
        """Test handling of paywall content."""
        markdown = """# Premium Content

## Log In Required

You need to log in to view this content.

## Get Full Access

Subscribe for unlimited access.
"""
        result = processor.refine_article(markdown)

        # Paywall content should return empty or minimal content
        assert len(result) < 50  # Very short result expected

    def test_refine_no_headings(self, processor):
        """Test refinement of content with no headings."""
        markdown = "This is just plain text with no structure."

        result = processor.refine_article(markdown)

        # Should return original content when no headings found
        assert result == markdown

    def test_refine_preserve_title(self, processor):
        """Test that title preservation works correctly."""
        markdown = """# Main Article Title

## Abstract

This is the abstract content.

## Introduction

This is the introduction.
"""
        result = processor.refine_article(markdown)

        # Should preserve the title before abstract
        assert "Main Article Title" in result
        assert "Abstract" in result


class TestDuplicateRemoval:
    """Test duplicate content removal functionality."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_remove_duplicate_sections(self, processor):
        """Test removal of duplicate sections."""
        markdown = """# Title 1

## Abstract

First abstract content.

# Title 2

## Abstract

This is the real abstract content.

## Introduction

Main content here.
"""
        result = processor.refine_article(markdown)

        # Should remove the first duplicate section
        assert result.count("Abstract") == 1
        assert "real abstract content" in result
        assert "First abstract content" not in result

    def test_remove_duplicate_h1_titles(self, processor):
        """Test removal of duplicate H1 titles."""
        markdown = """# Article Title

Some preamble content.

# Article Title

## Abstract

The real content starts here.
"""
        result = processor.refine_article(markdown)

        # Should start from second title occurrence
        assert "preamble content" not in result
        assert "real content" in result


class TestAcademicStructureDetection:
    """Test academic structure detection."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_has_academic_structure_positive(self, processor):
        """Test detection of academic structure."""
        headings = [
            HeadingInfo("Abstract", 2, 0, 10, "", is_relevant=True),
            HeadingInfo("Introduction", 2, 20, 30, "", is_relevant=True),
            HeadingInfo("Methods", 2, 40, 50, "", is_relevant=True),
            HeadingInfo("Results", 2, 60, 70, "", is_relevant=True),
        ]

        assert processor._has_academic_structure(headings) is True

    def test_has_academic_structure_negative(self, processor):
        """Test when content lacks academic structure."""
        headings = [
            HeadingInfo("Random Section", 2, 0, 10, "", is_relevant=False),
            HeadingInfo("Another Section", 2, 20, 30, "", is_relevant=False),
        ]

        assert processor._has_academic_structure(headings) is False

    def test_has_academic_structure_minimal(self, processor):
        """Test minimal academic structure detection."""
        headings = [
            HeadingInfo("Abstract", 2, 0, 10, "", is_relevant=True),
            HeadingInfo("Introduction", 2, 20, 30, "", is_relevant=True),
        ]

        # Should detect structure with 2+ academic terms
        assert processor._has_academic_structure(headings) is True


class TestInlineChromeCleanup:
    """Test inline chrome cleanup functionality."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_cleanup_inline_chrome(self, processor):
        """Test removal of inline chrome elements."""
        text = """# Main Content

This is academic content.

Sign in to continue reading.

Follow us on Twitter.

More academic content here.

Subscribe to newsletter.
"""
        result = processor._cleanup_inline_chrome(text)

        # Should preserve academic content
        assert "Main Content" in result
        assert "academic content" in result
        # Should remove chrome elements
        assert "Sign in" not in result
        assert "Follow us" not in result
        assert "Subscribe" not in result

    def test_cleanup_irrelevant_headings_skip_sections(self, processor):
        """Test that irrelevant headings cause their sections to be skipped."""
        text = """# Main Article

Good content here.

## References

Citation 1
Citation 2

## More Good Content

This should be preserved.
"""
        result = processor._cleanup_inline_chrome(text)

        # Should preserve main content
        assert "Main Article" in result
        assert "Good content here" in result
        assert "More Good Content" in result
        # Should skip references section content
        assert "Citation 1" not in result
        assert "Citation 2" not in result

    def test_cleanup_short_chrome_lines(self, processor):
        """Test removal of short chrome lines."""
        text = """# Article

Real content paragraph that is long enough to not be considered chrome.

Log in

password

This is more real content that should be preserved.
"""
        result = processor._cleanup_inline_chrome(text)

        # Should preserve long content
        assert "Real content paragraph" in result
        assert "more real content" in result
        # Should remove short chrome lines
        assert "Log in" not in result
        assert "password" not in result


class TestBoundaryDetection:
    """Test core content boundary detection."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_find_core_boundaries_with_abstract(self, processor):
        """Test finding core boundaries starting with abstract."""
        markdown = """# Title

Some preamble.

## Abstract

Abstract content.

## Introduction

Intro content.

## References

Ref content.
"""
        headings = processor._extract_and_classify_headings(markdown)
        relevant = [h for h in headings if h.is_relevant]

        start, end = processor._find_core_boundaries(markdown, headings, relevant)

        # Should start from title (preserving title before abstract)
        title_pos = markdown.find("# Title")
        abstract_pos = markdown.find("## Abstract")
        refs_pos = markdown.find("## References")

        # Start should be at title, end should be before references
        assert start <= title_pos
        assert end == refs_pos

    def test_find_core_boundaries_no_title(self, processor):
        """Test finding boundaries when there's no preceding title."""
        markdown = """## Introduction

Intro content.

## Methods  

Methods content.

## References

Ref content.
"""
        headings = processor._extract_and_classify_headings(markdown)
        relevant = [h for h in headings if h.is_relevant]

        start, end = processor._find_core_boundaries(markdown, headings, relevant)

        intro_pos = markdown.find("## Introduction")
        refs_pos = markdown.find("## References")

        # Should start at introduction, end before references
        assert start == intro_pos
        assert end == refs_pos


class TestUtilityFunctions:
    """Test utility and helper functions."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_find_preceding_larger_heading(self, processor):
        """Test finding preceding larger (lower level number) heading."""
        headings = [
            HeadingInfo("Title", 1, 0, 10, ""),
            HeadingInfo("Section", 2, 20, 30, ""),
            HeadingInfo("Subsection", 3, 40, 50, ""),
        ]

        # Find heading before subsection that's larger (lower level)
        result = processor._find_preceding_larger_heading(headings, headings[2])

        # Should find the level 2 section
        assert result == headings[1]

    def test_find_preceding_larger_heading_none(self, processor):
        """Test when no preceding larger heading exists."""
        headings = [
            HeadingInfo("Title", 1, 0, 10, ""),
        ]

        result = processor._find_preceding_larger_heading(headings, headings[0])

        assert result is None

    def test_find_next_same_or_higher_level_heading(self, processor):
        """Test finding next heading at same or higher level."""
        headings = [
            HeadingInfo("Section", 2, 0, 10, ""),
            HeadingInfo("Subsection", 3, 20, 30, ""),
            HeadingInfo("Another Section", 2, 40, 50, ""),
        ]

        # From first heading, should find third (same level)
        result = processor._find_next_same_or_higher_level_heading(
            headings, headings[0]
        )

        assert result == headings[2].start_pos

    def test_find_next_same_or_higher_level_heading_none(self, processor):
        """Test when no next same/higher level heading exists."""
        headings = [
            HeadingInfo("Section", 2, 0, 10, ""),
            HeadingInfo("Subsection", 3, 20, 30, ""),
        ]

        result = processor._find_next_same_or_higher_level_heading(
            headings, headings[0]
        )

        assert result is None


class TestLegacyFunctions:
    """Test legacy backward compatibility functions."""

    def test_extract_headings_function(self):
        """Test legacy extract_headings function."""
        markdown = """# Title

## Introduction

Content here.

## References

Ref content.
"""
        headings = extract_headings(markdown)

        assert len(headings) == 3
        assert headings[0]["text"] == "Title"
        assert headings[0]["level"] == 1
        assert headings[1]["text"] == "Introduction"
        assert headings[1]["is_relevant"] is True
        assert headings[2]["text"] == "References"
        assert headings[2]["is_irrelevant"] is True

    def test_classify_heading_relevance_function(self):
        """Test legacy classify_heading_relevance function."""
        # Test relevant heading
        result = classify_heading_relevance("Abstract")
        assert result["is_relevant"] is True
        assert result["is_irrelevant"] is False
        assert result["is_paywall"] is False

        # Test irrelevant heading
        result = classify_heading_relevance("References")
        assert result["is_relevant"] is False
        assert result["is_irrelevant"] is True
        assert result["is_paywall"] is False

        # Test paywall heading
        result = classify_heading_relevance("Log In")
        assert result["is_relevant"] is False
        assert result["is_irrelevant"] is True
        assert result["is_paywall"] is True

    def test_refine_article_content_function(self):
        """Test legacy refine_article_content function."""
        markdown = """# Title

## Abstract

Abstract content.

## References

Ref content.
"""
        result = refine_article_content(markdown)

        # Should behave same as ContentProcessor.refine_article
        processor = ContentProcessor()
        expected = processor.refine_article(markdown)

        assert result == expected


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_over_trim_protection(self, processor):
        """Test that over-trimming protection works."""
        # Create content where aggressive trimming would remove too much
        short_content = "# Title\n\n## Some Section\n\nBrief content."

        result = processor.refine_article(short_content)

        # Should not over-trim short content
        assert len(result) >= len(short_content) * 0.5  # At least 50% preserved

    def test_malformed_headings(self, processor):
        """Test handling of malformed headings."""
        markdown = """###Malformed heading without space
        
## Proper heading

###### Six levels deep

Regular content.
"""
        # Should handle gracefully without errors
        headings = processor._extract_and_classify_headings(markdown)
        result = processor.refine_article(markdown)

        assert len(headings) >= 1  # Should extract proper heading
        assert isinstance(result, str)

    def test_unicode_content(self, processor):
        """Test handling of unicode content."""
        markdown = """# Título del Artículo

## Résumé 

Contenu en français avec des accents.

## Методы

Содержание на русском языке.

## 方法

中文内容测试。
"""
        result = processor.refine_article(markdown)

        # Should handle unicode without errors
        assert isinstance(result, str)
        assert len(result) > 0

    def test_very_long_headings(self, processor):
        """Test handling of very long headings."""
        long_heading = "A" * 500  # 500 character heading
        markdown = f"""# {long_heading}

Regular content follows.
"""
        headings = processor._extract_and_classify_headings(markdown)

        assert len(headings) == 1
        assert headings[0].text == long_heading

    def test_empty_sections(self, processor):
        """Test handling of empty sections."""
        markdown = """# Title

## Abstract

## Introduction

Some actual content.

## Methods

## Results

More content.
"""
        result = processor.refine_article(markdown)

        # Should handle empty sections without errors
        assert isinstance(result, str)
        assert "actual content" in result
        assert "More content" in result


class TestConstantsAndConfiguration:
    """Test that constants and configuration values are reasonable."""

    def test_relevant_headings_not_empty(self):
        """Test that relevant headings set is not empty."""
        assert len(RELEVANT_HEADINGS) > 0
        assert "abstract" in RELEVANT_HEADINGS
        assert "introduction" in RELEVANT_HEADINGS

    def test_irrelevant_headings_not_empty(self):
        """Test that irrelevant headings set is not empty."""
        assert len(IRRELEVANT_HEADINGS) > 0
        assert "references" in IRRELEVANT_HEADINGS
        assert "acknowledgments" in IRRELEVANT_HEADINGS

    def test_paywall_patterns_not_empty(self):
        """Test that paywall patterns are defined."""
        assert len(PAYWALL_PATTERNS) > 0
        # Check that patterns compile correctly
        import re

        for pattern in PAYWALL_PATTERNS:
            re.compile(pattern)  # Should not raise exception

    def test_chrome_keywords_not_empty(self):
        """Test that chrome keywords are defined."""
        assert len(CHROME_INLINE_KEYWORDS) > 0
        assert "sign in" in CHROME_INLINE_KEYWORDS
        assert "password" in CHROME_INLINE_KEYWORDS

    def test_threshold_values_reasonable(self):
        """Test that threshold values are reasonable."""
        assert 0 < OVER_TRIM_THRESHOLD < 1  # Should be percentage
        assert INLINE_SHORT_LINE_MAXLEN > 0  # Should be positive
        assert INLINE_SHORT_LINE_MAXLEN < 1000  # Should be reasonable limit
