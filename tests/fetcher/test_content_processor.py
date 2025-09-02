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

    def test_extract_split_headings(self, processor):
        """Test extraction of split headings (hash marks on one line, text on next)."""
        markdown = """# Normal Title

## 
REFERENCES

Some content.

## 
CITING LITERATURE

More content.

### Normal Section
"""
        headings = processor._extract_and_classify_headings(markdown)

        assert len(headings) == 4
        assert headings[0].text == "Normal Title"
        assert headings[0].level == 1
        assert headings[1].text == "REFERENCES"
        assert headings[1].level == 2
        assert headings[2].text == "CITING LITERATURE"
        assert headings[2].level == 2
        assert headings[3].text == "Normal Section"
        assert headings[3].level == 3


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


class TestRegressionCases:
    """Test regression cases from real-world examples."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_cox5a_structure_with_references_not_removed(self, processor):
        """Test structure that causes title loss and references preservation."""
        markdown = """Site navigation
Privacy policy
Login required

# Paper Title

More site chrome
Cookie settings
Subscribe now

### Abstract

Abstract content here.

Some content between sections

### Graphical Abstract

Graphical abstract content here.

More content between sections

## 
REFERENCES

Reference content that should be removed.
Reference list continues here.
"""
        result = processor.refine_article(markdown)

        # Title should be preserved
        assert "Paper Title" in result

        # Academic content should be preserved
        assert "Abstract content" in result
        assert "Graphical abstract content" in result

        # Site chrome should be removed
        assert "Site navigation" not in result
        assert "Privacy policy" not in result
        assert "Subscribe now" not in result

        # References should be removed
        assert "Reference content that should be removed" not in result
        assert "Reference list continues" not in result

        # Formatting should be correct - proper spacing between title and abstract
        assert "# Paper Title\n\n### Abstract" in result
        assert result.count("# Paper Title") == 1  # No duplicated titles

    def test_preserve_major_headings_between_title_and_abstract(self, processor):
        """Test that major headings between title and abstract are preserved."""
        markdown = """# Paper Title

## Study Design

This section describes our experimental methodology and is important.

### Abstract

Abstract content here.

### Results

Results content here.
"""
        result = processor.refine_article(markdown)

        # All content should be preserved because Study Design is a major heading (H2 >= H3)
        assert "Paper Title" in result
        assert "Study Design" in result
        assert "experimental methodology" in result
        assert "Abstract content" in result
        assert "Results content" in result

        # Formatting should preserve structure with Study Design intact
        lines = result.strip().split("\n")
        assert lines[0] == "# Paper Title"
        assert "## Study Design" in result
        assert "### Abstract" in result

    def test_skip_minor_headings_between_title_and_abstract(self, processor):
        """Test that minor headings between title and abstract are skipped."""
        markdown = """# Paper Title

#### Author Information

Contact details and affiliations here.

### Abstract

Abstract content here.
"""
        result = processor.refine_article(markdown)

        # Gap should be removed because Author Information is minor (H4 < H3)
        assert "Paper Title" in result
        assert "Author Information" not in result
        assert "Contact details" not in result
        assert "Abstract content" in result

        # Formatting should have proper spacing after gap removal
        assert "# Paper Title\n\n### Abstract" in result

    def test_remove_journal_footer_chrome_cascade(self, processor):
        """Test that journal footer sections are removed in cascade after first irrelevant heading."""
        markdown = """# Paper Title

### Abstract
Academic content about the research.

### Methods
Research methodology here.

## Article metrics
## Related Articles
View abstract
Open in viewer

## Articles & Issues
  * Articles in Press
  * Current IssueE 
  * List of Issues
  * Supplements

## Collections
  * All Collections
  * Shared Science

## For Authors
  * Instructions for Authors
  * Submit Your Manuscript

## AAAAI
  * AAAAI Information
  * AAAAI Website

## Follow Us
  * Facebook
  * Twitter
"""
        result = processor.refine_article(markdown)

        # Academic content should be preserved
        assert "Paper Title" in result
        assert "Academic content about the research" in result
        assert "Research methodology here" in result

        # All footer chrome should be removed after first irrelevant heading
        assert "Article metrics" not in result
        assert "Related Articles" not in result
        assert "Articles & Issues" not in result
        assert "Collections" not in result
        assert "For Authors" not in result
        assert "AAAAI" not in result
        assert "Follow Us" not in result
        assert "Articles in Press" not in result
        assert "Submit Your Manuscript" not in result

    def test_remove_author_section_gap_between_title_and_abstract(self, processor):
        """Test that author sections between title and abstract are removed."""
        markdown = """# Research Paper Title

John Smith
John Smith
Affiliations
Department of Biology, University of Science, City, Country
Search for articles by this author
1 ∙ Jane Doe
Jane Doe
Affiliations  
Department of Chemistry, University of Research, City, Country
Search for articles by this author
2 ∙ Bob Johnson
Bob Johnson
Correspondence
Correspondence and requests for materials should be addressed to Bob Johnson
bob.johnson@university.edu
Affiliations
Department of Physics, University of Technology, City, Country
Article Info
Publication History:
Received October 20, 2022; Revised January 24, 2023
DOI: 10.1016/j.example.2023.100798
Download PDF
Outline
  * Abstract
  * Introduction
  * Methods

## Abstract
### Purpose
This study investigates important biological mechanisms.
### Methods  
We used advanced techniques to analyze the data.

## Introduction
The field has long been interested in this topic.
"""
        result = processor.refine_article(markdown)

        # Title should be preserved
        assert "Research Paper Title" in result

        # Abstract content should be preserved
        assert "This study investigates important biological mechanisms" in result
        assert "We used advanced techniques to analyze the data" in result

        # Introduction should be preserved
        assert "The field has long been interested in this topic" in result

        # Author section should be removed
        assert "John Smith" not in result
        assert "Jane Doe" not in result
        assert "Bob Johnson" not in result
        assert "Department of Biology" not in result
        assert "Search for articles by this author" not in result
        assert "Correspondence and requests for materials" not in result
        assert "bob.johnson@university.edu" not in result
        assert "Publication History" not in result
        assert "Download PDF" not in result
        assert "Outline" not in result

        # Proper formatting should be maintained
        assert "# Research Paper Title\n\n## Abstract" in result

    def test_prefer_abstract_over_early_navigation_headings(self, processor):
        """Test that Abstract is preferred over earlier navigation headings marked as relevant."""
        markdown = """#### Important Guidelines Document
This is a navigation heading that might get marked as relevant.

#### Standards for Interpretation
Another navigation heading.

# Real Paper Title

Author Name
Author Name
Affiliations
University Department
Search for articles by this author
More author content here
Publication details
Download PDF

## Abstract
This is the actual paper abstract content.

## Introduction  
This is the introduction section.
"""
        result = processor.refine_article(markdown)

        # Real content should be preserved
        assert "Real Paper Title" in result
        assert "This is the actual paper abstract content" in result
        assert "This is the introduction section" in result

        # Navigation headings should be removed
        assert "Important Guidelines Document" not in result
        assert "Standards for Interpretation" not in result

        # Author section should be removed
        assert "Author Name" not in result
        assert "University Department" not in result
        assert "Search for articles by this author" not in result
        assert "Publication details" not in result
        assert "Download PDF" not in result

        # Should jump directly from title to abstract
        assert "# Real Paper Title\n\n## Abstract" in result

    def test_remove_same_level_unknown_headings_with_gap_removal(self, processor):
        """Test that unknown headings at same level as first academic heading are removed with gap."""
        markdown = """# Research Paper Title

## Significance Statement
This work provides important insights into biological mechanisms
and represents a significant advance in our understanding.

Author Name
Author Name  
Affiliations
Department info

## Abstract
This is the abstract content.

## Introduction
This is the introduction.
"""
        result = processor.refine_article(markdown)

        # Title should be preserved
        assert "Research Paper Title" in result

        # Abstract and introduction should be preserved
        assert "This is the abstract content" in result
        assert "This is the introduction" in result

        # Same-level unknown heading and gap should be removed (H2 is not > H2)
        assert "Significance Statement" not in result
        assert (
            "This work provides important insights into biological mechanisms"
            not in result
        )
        assert "significant advance in our understanding" not in result
        assert "Author Name" not in result
        assert "Department info" not in result

    def test_preserve_more_important_unknown_headings_block_gap_removal(
        self, processor
    ):
        """Test that more important unknown headings between title and abstract prevent gap removal."""
        markdown = """# Research Paper Title

Author Name
Department info

## Editorial Commentary
This section provides editorial commentary on the research
and gives additional context that should be preserved.

More author content
Publication details

### Abstract
This is the abstract content.
"""
        result = processor.refine_article(markdown)

        # Everything should be preserved since there's a more important (H2) heading in the gap (vs H3 Abstract)
        assert "Research Paper Title" in result
        assert "Editorial Commentary" in result
        assert "This section provides editorial commentary on the research" in result
        assert "and gives additional context that should be preserved" in result
        assert "This is the abstract content" in result

        # Even author content should be preserved since gap removal was blocked
        assert "Author Name" in result
        assert "Department info" in result
        assert "More author content" in result

    def test_metadata_headings_allow_gap_removal(self, processor):
        """Test that metadata headings like 'Contributed equally' don't block gap removal."""
        markdown = """# Paper Title

Author information here
Department affiliations

# Contributed equally.
Additional metadata

## Affiliations
* University A
* University B

Author notes and details

## Abstract
This is the abstract content.

## Introduction
This is the introduction.
"""
        result = processor.refine_article(markdown)

        # Title and academic content should be preserved
        assert "Paper Title" in result
        assert "This is the abstract content" in result
        assert "This is the introduction" in result

        # Metadata content should be removed via gap removal
        assert "Author information here" not in result
        assert "Department affiliations" not in result
        assert "Contributed equally" not in result
        assert "Additional metadata" not in result
        assert "University A" not in result
        assert "Author notes and details" not in result

        # Should jump directly from title to abstract
        assert "# Paper Title\n\n## Abstract" in result


class TestFormattingValidation:
    """Test proper markdown formatting and structure preservation."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_proper_heading_spacing_after_smart_extraction(self, processor):
        """Test that smart gap removal produces proper markdown spacing."""
        markdown = """Site chrome
Login prompts

# Research Paper Title

Author: John Doe
Email: john@example.com
Affiliation: University

### Abstract

This is the abstract content with important research findings.

### Methods

Methodology section content here.

## References

Reference 1
Reference 2
"""
        result = processor.refine_article(markdown)

        # Check content is preserved
        assert "Research Paper Title" in result
        assert "abstract content" in result
        assert "Methodology section" in result

        # Check formatting is correct
        lines = [line for line in result.split("\n") if line.strip()]  # Non-empty lines

        # Title should be first non-empty line
        assert lines[0] == "# Research Paper Title"

        # Abstract should follow with proper spacing
        assert "# Research Paper Title\n\n### Abstract" in result

        # No author metadata should remain
        assert "Author: John Doe" not in result
        assert "Email:" not in result
        assert "Affiliation:" not in result

    def test_no_concatenated_headings(self, processor):
        """Test that headings are never concatenated without proper line breaks."""
        markdown = """# Title
Author info
### Abstract
Content
### Results  
More content
"""
        result = processor.refine_article(markdown)

        # Check for proper line breaks between all headings
        assert "Title### Abstract" not in result  # No concatenation
        assert "Abstract### Results" not in result  # No concatenation

        # Check proper spacing exists
        assert "# Title\n\n### Abstract" in result

    def test_preserve_section_spacing(self, processor):
        """Test that spacing between academic sections is preserved."""
        markdown = """# Paper Title

### Abstract

Abstract content here.

Some additional abstract details.

### Introduction

Introduction content here.

### Methods

Methods content here.
"""
        result = processor.refine_article(markdown)

        # Verify proper structure is maintained
        lines = result.split("\n")

        # Find key sections
        title_idx = next(i for i, line in enumerate(lines) if line == "# Paper Title")
        abstract_idx = next(i for i, line in enumerate(lines) if line == "### Abstract")
        intro_idx = next(
            i for i, line in enumerate(lines) if line == "### Introduction"
        )

        # Check spacing: should have blank line before each major section
        assert lines[abstract_idx - 1] == ""  # Blank line before Abstract
        assert lines[intro_idx - 1] == ""  # Blank line before Introduction

    def test_clean_markdown_structure(self, processor):
        """Test that result is valid markdown with clean structure."""
        markdown = """# Title
Chrome content
### Abstract
Abstract content
### Results
Results content
## References
Ref content
"""
        result = processor.refine_article(markdown)

        # Should start and end cleanly
        assert not result.startswith("\n")  # No leading newlines
        assert not result.endswith("\n\n")  # No excessive trailing newlines

        # No double headings or malformed structure
        assert result.count("# Title") == 1
        assert "##" not in result or "References" not in result  # References removed

        # Headings should be properly formatted
        for line in result.split("\n"):
            if line.startswith("#"):
                # Should have space after hash
                assert (
                    line.startswith("# ")
                    or line.startswith("## ")
                    or line.startswith("### ")
                )


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


class TestMinimalAcademicStructure:
    """Test handling of documents with minimal academic structure."""

    def test_single_abstract_heading_accepted(self):
        """Test that documents with only an Abstract heading are processed."""
        markdown = """
# Navigation Chrome
Skip to main content
Login | Sign up

# Paper Title
Authors: John Doe, Jane Smith

## Abstract
This is the abstract content describing the research.
More abstract content here.

# References
1. Citation 1
2. Citation 2

# Footer Chrome
Privacy Policy | Terms of Service
        """
        processor = ContentProcessor()
        result = processor.refine_article(markdown)

        # Should extract the academic content, not return original
        assert len(result) < len(markdown)  # Should be trimmed
        assert "Paper Title" in result
        assert "This is the abstract content" in result
        assert "Skip to main content" not in result  # Chrome removed
        assert "Privacy Policy" not in result  # Footer removed

    def test_abstract_only_has_academic_structure(self):
        """Test that _has_academic_structure accepts abstract-only documents."""
        markdown = "## Abstract\nThis is an abstract."
        processor = ContentProcessor()
        headings = processor._extract_and_classify_headings(markdown)

        assert processor._has_academic_structure(headings) == True

    def test_figures_only_has_academic_structure(self):
        """Test that documents with only figures heading are accepted."""
        markdown = "## Figures\nFigure 1: Description"
        processor = ContentProcessor()
        headings = processor._extract_and_classify_headings(markdown)

        assert processor._has_academic_structure(headings) == True

    def test_two_relevant_headings_accepted(self):
        """Test that documents with 2+ relevant headings are accepted."""
        markdown = """
## Results
Some results here.

## Methods  
Some methods here.
        """
        processor = ContentProcessor()
        headings = processor._extract_and_classify_headings(markdown)

        assert processor._has_academic_structure(headings) == True


class TestContentPreservation:
    """Test that important academic content is not accidentally removed."""

    def test_preserve_methods_section_with_unusual_formatting(self):
        """Test that methods sections with unusual formatting are preserved."""
        markdown = """# Comprehensive Research Study on Cellular Biology

## Abstract
This study investigates cellular mechanisms through advanced microscopy techniques.
We examined protein interactions and metabolic pathways in living cells.
Our findings reveal novel insights into cellular function and disease states.

## Methods and Materials
We utilized advanced fluorescence microscopy and biochemical assays.
Standard cell culture protocols were followed for all experiments.
Protein expression was quantified using Western blot analysis.

## Methodology 
Alternative methods heading for statistical analysis approaches.
We applied both parametric and non-parametric statistical tests.
Data visualization was performed using advanced software packages.

## Experimental Procedures
Another methods variant describing detailed protocols.
Each experiment was repeated three times for statistical significance.
Quality control measures were implemented throughout the study.

## Results
Significant changes in protein expression were observed.
Statistical analysis confirmed the reliability of our findings.

## References
Citation 1: Previous study on cellular mechanisms
Citation 2: Methodology reference paper
        """
        processor = ContentProcessor()
        result = processor.refine_article(markdown)

        assert "Methods and Materials" in result
        assert "Methodology" in result
        assert "Experimental Procedures" in result
        assert "advanced fluorescence microscopy" in result
        assert "statistical analysis approaches" in result
        assert "detailed protocols" in result

    def test_preserve_results_section_variants(self):
        """Test that various result section formats are preserved."""
        markdown = """# Advanced Proteomics Analysis Study

## Abstract
This comprehensive study examines protein expression patterns in diseased tissue.
We utilized mass spectrometry and bioinformatics analysis for protein identification.
Our research provides new insights into disease mechanisms and therapeutic targets.

## Results
Main results here showing significant protein expression changes.
Statistical significance was achieved across multiple protein families.
Novel biomarkers were identified through systematic analysis.

## Results and Discussion
Combined section presenting both findings and their interpretation.
Our data suggests important regulatory mechanisms are disrupted.
These findings have implications for therapeutic intervention strategies.

## Experimental Results
Detailed results from controlled laboratory experiments.
Multiple independent experiments confirmed reproducibility of findings.
Quantitative analysis revealed consistent patterns across sample groups.

## Key Findings
Important findings that advance our understanding of disease biology.
Three major protein pathways were significantly altered.
Clinical relevance of these changes was demonstrated through validation studies.

## References
Citations for methodology and previous research
        """
        processor = ContentProcessor()
        result = processor.refine_article(markdown)

        assert "significant protein expression changes" in result
        assert "Combined section presenting both findings" in result
        assert "controlled laboratory experiments" in result
        assert "advance our understanding" in result

    def test_preserve_conclusion_section_variants(self):
        """Test that conclusion sections with different names are preserved."""
        markdown = """# Clinical Trial Results for Novel Therapeutic Intervention

## Abstract
This randomized controlled trial evaluated a new therapeutic approach.
We recruited 500 patients across multiple clinical centers.
Primary endpoints showed statistically significant improvement in patient outcomes.

## Conclusion
Main conclusion summarizing the clinical significance of our findings.
The novel therapeutic approach demonstrated superior efficacy compared to standard care.
Patient safety profiles were acceptable with minimal adverse events.

## Conclusions
Multiple conclusions derived from comprehensive analysis of trial data.
Both primary and secondary endpoints supported therapeutic benefit.
Cost-effectiveness analysis favored the new treatment approach.

## Summary and Conclusions
Combined summary highlighting key achievements and clinical implications.
This research represents a significant advance in patient care.
Implementation guidelines for clinical practice are provided.

## Discussion and Conclusions
Discussion with conclusions integrating findings with existing literature.
Our results support broader adoption of this therapeutic strategy.
Future research directions are outlined based on current findings.

## References
Clinical trial methodology references
Previous therapeutic studies
        """
        processor = ContentProcessor()
        result = processor.refine_article(markdown)

        assert "clinical significance of our findings" in result
        assert "comprehensive analysis of trial data" in result
        assert "significant advance in patient care" in result
        assert "integrating findings with existing literature" in result

    def test_preserve_appendix_content(self):
        """Test that appendix sections are preserved as academic content."""
        markdown = """# Longitudinal Study of Environmental Impact Assessment

## Abstract
This longitudinal study tracked environmental changes over a decade.
We analyzed multiple ecosystem parameters using standardized monitoring protocols.
Our findings document significant environmental trends and their underlying causes.

## Methods
Standard environmental monitoring protocols were implemented across study sites.
Data collection included water quality, soil composition, and biodiversity metrics.
Statistical analysis employed time-series modeling for trend detection.

## Appendix A: Supplementary Methods
Additional methods for specialized sample collection and analysis procedures.
Detailed protocols for rare species identification and habitat assessment.
Quality assurance procedures for long-term data consistency.

## Appendices
Multiple appendices containing comprehensive data tables and statistical outputs.
Extended methodology descriptions for complex analytical procedures.
Supplementary figures illustrating temporal trends across study parameters.

## Appendix: Statistical Analysis
Statistical details including model selection criteria and validation procedures.
Comprehensive description of time-series analysis methods.
Sensitivity analysis results for key model parameters.

## References
Environmental monitoring methodology references
Statistical analysis software documentation
        """
        processor = ContentProcessor()
        result = processor.refine_article(markdown)

        assert "specialized sample collection" in result
        assert "comprehensive data tables" in result
        assert "time-series analysis methods" in result

    def test_dont_remove_legitimate_author_content(self):
        """Test that author-related academic content is not removed."""
        markdown = """# Multi-Institutional Collaborative Research Project

## Abstract
This collaborative study involved researchers from multiple institutions.
We investigated complex biological systems through interdisciplinary approaches.
Significant discoveries were made through coordinated research efforts.

## Author Contributions
John Smith designed experiments and performed statistical analysis.
Jane Doe conducted laboratory experiments and data collection.
Bob Johnson provided bioinformatics expertise and computational analysis.
All authors contributed to manuscript preparation and revision.

## Data Availability Statement
All research data is available at institutional repository XYZ.
Raw data files and analysis scripts can be accessed through secure portal.
Metadata documentation follows international standards for data sharing.

## Acknowledgments
We thank the funding agencies for supporting this research initiative.
Technical assistance from laboratory staff was essential for project success.

## References
Methodology references and previous studies
        """
        processor = ContentProcessor()
        result = processor.refine_article(markdown)

        # Academic author contributions should be preserved
        assert "Author Contributions" in result
        assert "designed experiments and performed" in result
        assert "Data Availability Statement" in result
        assert "institutional repository" in result

        # Acknowledgments should be removed as irrelevant
        assert "We thank the funding agencies" not in result


class TestPubMedSpecificCases:
    """Test specific PubMed document patterns that have caused issues."""

    def test_pubmed_duplicate_titles_with_metadata(self):
        """Test PubMed documents with duplicate titles separated by metadata."""
        markdown = """
# Navigation Chrome
Skip to main content

# Paper Title
Authors: John Doe, Jane Smith

# Contributed equally
PMID: 12345
DOI: 10.1000/test

# Paper Title  
Short citation format

## Abstract
This is the actual abstract content.

## Figures
Figure descriptions here.

# Footer Chrome
Privacy Policy
        """
        processor = ContentProcessor()
        result = processor.refine_article(markdown)

        # Should start from the second title (closest to content)
        assert result.startswith("# Paper Title")
        assert "This is the actual abstract content" in result
        assert "Figure descriptions here" in result

        # Chrome should be removed
        assert "Skip to main content" not in result
        assert "Privacy Policy" not in result

        # Metadata should be cleaned up
        assert "PMID: 12345" not in result or result.count("PMID: 12345") == 1

    def test_abstract_only_paper_extraction(self):
        """Test papers with only abstract and no full text."""
        markdown = """
# Website Navigation
Home | Search | Help

# Paper Title
Author list with affiliations

## Abstract
This is a complete abstract with background, methods, results, and conclusions.
The paper may not have full text available, just the abstract.

# Website Footer
Terms | Privacy | Contact

# Related Articles
List of related papers
        """
        processor = ContentProcessor()
        result = processor.refine_article(markdown)

        # Should extract just title and abstract
        assert "Paper Title" in result
        assert "This is a complete abstract" in result
        assert "just the abstract" in result

        # Website chrome should be removed
        assert "Website Navigation" not in result
        assert "Website Footer" not in result
        assert "Related Articles" not in result

    def test_figures_only_paper_extraction(self):
        """Test papers with only figures section (like figure galleries)."""
        markdown = """
# Site Header
Navigation links

# Figure Gallery: Study Results

## Figures
Figure 1: Important results chart
Figure 2: Additional data visualization
Figure 3: Summary diagram

# Site Footer
Copyright notice
        """
        processor = ContentProcessor()
        result = processor.refine_article(markdown)

        # Should preserve figure content
        assert "Figure Gallery: Study Results" in result
        assert "Important results chart" in result
        assert "Additional data visualization" in result
        assert "Summary diagram" in result

        # Site chrome should be removed
        assert "Site Header" not in result
        assert "Navigation links" not in result
        assert "Site Footer" not in result


class TestOverTrimProtection:
    """Test over-trim protection mechanisms."""

    def test_over_trim_protection_with_valid_academic_content(self):
        """Test that over-trim protection doesn't kick in for valid academic papers."""
        short_academic = """
# Title

## Abstract
Short but valid academic abstract.

## Methods
Brief methods.
        """
        processor = ContentProcessor()
        result = processor.refine_article(short_academic)

        # Should process normally despite being short
        assert "Title" in result
        assert "Short but valid academic abstract" in result
        assert "Brief methods" in result

    def test_over_trim_protection_with_non_academic_content(self):
        """Test that over-trim protection preserves non-academic content when appropriate."""
        non_academic = """
# Welcome to Our Website

## Latest News
Recent updates about our company.

## Contact Us
Get in touch with our team.

## Privacy Policy
Our privacy policy details.
        """
        processor = ContentProcessor()
        result = processor.refine_article(non_academic)

        # Should preserve original content due to lack of academic structure
        assert len(result) == len(non_academic.strip())


class TestEdgeCaseProtection:
    """Test protection against edge cases that could break content extraction."""

    def test_empty_sections_dont_break_processing(self):
        """Test that empty sections don't break the content processor."""
        markdown = """
# Title

## Abstract

## Methods

Some content after empty sections.

## Results


## Conclusion
Final thoughts.
        """
        processor = ContentProcessor()
        result = processor.refine_article(markdown)

        # Should handle empty sections gracefully
        assert isinstance(result, str)
        assert "Title" in result
        assert "Some content after empty sections" in result
        assert "Final thoughts" in result

    def test_malformed_heading_recovery(self):
        """Test recovery from malformed headings."""
        markdown = """
# Valid Title

## Abstract
Good abstract content.

###Malformed Heading No Space
Content under malformed heading.

## Methods
Valid methods section.

# Another Bad Heading#Extra Hash
More content.
        """
        processor = ContentProcessor()
        result = processor.refine_article(markdown)

        # Should preserve good content and handle malformed headings gracefully
        assert "Valid Title" in result
        assert "Good abstract content" in result
        assert "Valid methods section" in result

    def test_unicode_and_special_characters(self):
        """Test handling of Unicode and special characters in academic content."""
        markdown = """
# Título del Artículo: α-β Analysis

## Abstract
This study examines α-particle interactions with β-radiation.
Temperature was measured in °C. Concentrations in μg/mL.

## Résultats
Les résultats montrent une corrélation significative.

## 方法
Chinese text in methods section.

## Заключение  
Russian conclusion text.
        """
        processor = ContentProcessor()
        result = processor.refine_article(markdown)

        # Should preserve Unicode content
        assert "Título del Artículo" in result
        assert "α-particle interactions" in result
        assert "°C" in result
        assert "μg/mL" in result
        assert "Les résultats" in result
        assert "Chinese text" in result or "方法" in result
        assert "Russian conclusion" in result or "Заключение" in result


class TestSpecificRegressionPrevention:
    """Test specific scenarios that have caused regressions."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_prevent_title_loss_with_references_preservation_bug(self, processor):
        """Regression test for COX5A paper bug: title lost, abstract lost, references kept."""
        markdown = """
Navigation | Privacy | Login

# Important Research Paper Title

Site chrome content
Subscribe to newsletter

### Abstract

This is crucial abstract content that must be preserved.
Contains important research findings and methodology.

### Results

Critical results section with data.

## 
REFERENCES

Reference 1: Should be removed
Reference 2: Should not appear in output
        """
        result = processor.refine_article(markdown)

        # Must preserve title and abstract
        assert "Important Research Paper Title" in result
        assert "This is crucial abstract content" in result
        assert "Contains important research findings" in result
        assert "Critical results section with data" in result

        # Must remove references and chrome
        assert "Reference 1: Should be removed" not in result
        assert "Reference 2: Should not appear" not in result
        assert "Subscribe to newsletter" not in result
        assert "Navigation | Privacy" not in result

    def test_prevent_unfiltered_document_return_bug(self, processor):
        """Regression test for bug where minimal structure documents returned unfiltered."""
        # Create a document that previously triggered the over-trim protection bug
        large_unfiltered = """
Home | About | Contact | Login | Register

Advanced Search Options:
- Search by author
- Search by keyword  
- Search by date range
- Search by journal

Navigation breadcrumb: Home > Literature > Papers > Details

Cookie Notice: This website uses cookies to enhance user experience.
Accept All | Reject All | Customize Settings

# Research Article: Gene Expression Analysis

Authors:
Dr. John Smith - Department of Biology, University A
Dr. Jane Doe - Department of Chemistry, University B
Dr. Bob Johnson - Department of Physics, University C

Corresponding Author: Dr. Bob Johnson (bob.johnson@universityc.edu)

Article Information:
Received: January 15, 2023
Accepted: March 22, 2023
Published: April 10, 2023
DOI: 10.1000/example.2023.001234

Download Options:
PDF | HTML | ePub | XML

Abstract

This study presents a comprehensive analysis of gene expression patterns
in response to environmental stress factors. The research utilized
state-of-the-art sequencing technology to examine transcriptional changes.

Key findings indicate significant upregulation of stress response genes
and downregulation of metabolic pathways under stress conditions.
These results provide important insights for therapeutic applications.

Article Metrics:
Views: 1,234 | Downloads: 567 | Citations: 89

Social Media:
Share on Twitter | Share on Facebook | Share on LinkedIn

Related Articles:
- Similar studies in this field
- Recent publications by these authors  
- Trending articles in this journal

Advertisement
Subscribe to our newsletter for latest updates!

Footer:
Terms of Use | Privacy Policy | Contact Us | About the Journal
Copyright 2023 Academic Publisher. All rights reserved.
        """
        result = processor.refine_article(large_unfiltered)

        # Should be significantly reduced, not return entire original
        assert len(result) < len(large_unfiltered) * 0.4  # Less than 40% of original

        # Should preserve academic content
        assert "Research Article: Gene Expression Analysis" in result
        assert "comprehensive analysis of gene expression" in result
        assert "Key findings indicate" in result

        # Should remove chrome
        assert "Cookie Notice:" not in result
        assert "Download Options:" not in result
        assert "Social Media:" not in result
        assert "Advertisement" not in result

    def test_prevent_aggressive_duplicate_removal_bug(self, processor):
        """Regression test for overly aggressive duplicate section removal."""
        markdown = """
# Paper Title

## Study Overview
This section provides important context about the research approach
and is distinct from the abstract despite similar content.

## Abstract
This study examines cellular mechanisms in disease progression.
Novel therapeutic targets are identified through analysis.

## Introduction
Building on the abstract, this section provides detailed background
including literature review and theoretical framework.

## Methods
Comprehensive methodology section with protocols.

## Results
Significant findings with statistical analysis.
        """
        result = processor.refine_article(markdown)

        # All academic content should be preserved
        assert "Study Overview" in result
        assert "important context about the research approach" in result
        assert "This study examines cellular mechanisms" in result
        assert "detailed background" in result
        assert "literature review" in result
        assert "Comprehensive methodology section" in result
        assert "Significant findings" in result

        # Content should not be over-trimmed
        assert len(result) > len(markdown) * 0.6  # At least 60% preserved

    def test_prevent_academic_heading_misclassification(self, processor):
        """Test that academic headings with unusual names are not misclassified."""
        markdown = """
# Research Paper

## Abstract
Standard abstract content.

## Experimental Design
Important methodology section.

## Statistical Analysis
Critical analysis methods.

## Data Availability
Important research data information.

## Author Contributions
Academic contribution statements.

## Competing Interests
Research ethics information.

## Supplementary Information
Additional research data.

# References
Citations to remove.
        """
        result = processor.refine_article(markdown)

        # All academic sections should be preserved
        assert "Experimental Design" in result
        assert "Statistical Analysis" in result
        assert "Data Availability" in result
        assert "Author Contributions" in result
        assert "Competing Interests" in result
        assert "Supplementary Information" in result

        # References should be removed
        assert "Citations to remove" not in result


class TestBoundaryConditionProtection:
    """Test boundary conditions that could cause filtering failures."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_single_word_headings(self, processor):
        """Test that single-word academic headings are preserved."""
        markdown = """
# Title

## Abstract
Abstract content here.

## Methods
Methods content.

## Results  
Results content.

## Discussion
Discussion content.

## References
Citations here.
        """
        result = processor.refine_article(markdown)

        assert "Abstract" in result
        assert "Methods" in result
        assert "Results" in result
        assert "Discussion" in result
        assert "Abstract content here" in result
        assert "Citations here" not in result

    def test_very_short_academic_content(self, processor):
        """Test that very short but valid academic content is preserved."""
        markdown = """
# Title

## Abstract
Short abstract.

## Methods
Brief method.

## Results
Key result.

## References
Ref 1
        """
        result = processor.refine_article(markdown)

        # Should preserve short academic content
        assert "Short abstract" in result
        assert "Brief method" in result
        assert "Key result" in result
        # Should remove references
        assert "Ref 1" not in result

    def test_nested_academic_structure(self, processor):
        """Test deeply nested academic sections are handled properly."""
        markdown = """
# Main Title

## Abstract
Top-level abstract.

### Background
Abstract background.

#### Study Rationale
Detailed rationale.

##### Specific Aims
Specific research aims.

###### Sub-objectives
Sub-level objectives.

## Methods
Main methods.

### Experimental Setup
Setup details.

#### Data Collection
Collection protocols.

##### Statistical Analysis
Analysis methods.

## References
Citations
        """
        result = processor.refine_article(markdown)

        # Should preserve nested academic structure
        assert "Top-level abstract" in result
        assert "Abstract background" in result
        assert "Detailed rationale" in result
        assert "Specific research aims" in result
        assert "Sub-level objectives" in result
        assert "Setup details" in result
        assert "Collection protocols" in result
        assert "Analysis methods" in result

        # Should remove references
        assert "Citations" not in result

    def test_mixed_relevant_irrelevant_same_level(self, processor):
        """Test handling when relevant and irrelevant headings are at same level."""
        markdown = """
# Paper Title

## Abstract
Academic content.

## Keywords
Keyword list - should this be removed?

## Introduction
More academic content.

## Acknowledgments
Should be removed.

## Methods
Important methods.

## References
Should be removed.
        """
        result = processor.refine_article(markdown)

        # Academic content should be preserved
        assert "Academic content" in result
        assert "More academic content" in result
        assert "Important methods" in result

        # Irrelevant sections should be removed
        assert "Acknowledgments" not in result
        assert "Should be removed" not in result

        # Keywords might be preserved as neutral (not explicitly irrelevant)
        # This is a judgment call - keeping the test flexible


class TestAcademicContentVariations:
    """Test variations in academic content that should be preserved."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_preserve_methodology_variants(self, processor):
        """Test that various methodology section names are preserved."""
        markdown = """
# Title

## Abstract
Abstract content.

## Methodology
Standard methodology.

## Materials and Methods
Combined section.

## Experimental Procedures
Procedure details.

## Study Design
Design information.

## Data Collection Methods
Collection details.

## Analytical Methods
Analysis procedures.

## References
Citations
        """
        result = processor.refine_article(markdown)

        # All method variants should be preserved
        assert "Standard methodology" in result
        assert "Combined section" in result
        assert "Procedure details" in result
        assert "Design information" in result
        assert "Collection details" in result
        assert "Analysis procedures" in result

    def test_preserve_analysis_section_variants(self, processor):
        """Test that analysis sections with different names are preserved."""
        markdown = """
# Title

## Abstract
Abstract content.

## Analysis
General analysis.

## Statistical Analysis  
Statistical methods.

## Data Analysis
Data processing.

## Bioinformatics Analysis
Computational analysis.

## Quantitative Analysis
Quantitative methods.

## Qualitative Analysis
Qualitative methods.

## References
Citations
        """
        result = processor.refine_article(markdown)

        # All analysis variants should be preserved
        assert "General analysis" in result
        assert "Statistical methods" in result
        assert "Data processing" in result
        assert "Computational analysis" in result
        assert "Quantitative methods" in result
        assert "Qualitative methods" in result

    def test_preserve_background_section_variants(self, processor):
        """Test that background sections are properly preserved."""
        markdown = """
# Title

## Abstract
Abstract content.

## Background
Study background.

## Literature Review
Previous research.

## Theoretical Framework
Theoretical basis.

## Study Rationale
Research rationale.

## Objectives
Research objectives.

## Hypothesis
Research hypothesis.

## References
Citations
        """
        result = processor.refine_article(markdown)

        # Academic background content should be preserved
        assert "Study background" in result
        assert "Previous research" in result
        assert "Theoretical basis" in result
        assert "Research rationale" in result
        assert "Research objectives" in result
        assert "Research hypothesis" in result


class TestConstantsAndConfiguration:
    """Test that constants and configuration values are reasonable."""

    def test_relevant_headings_not_empty(self):
        """Test that relevant headings set is not empty."""
        assert len(RELEVANT_HEADINGS) > 0
        assert "abstract" in RELEVANT_HEADINGS
        assert "introduction" in RELEVANT_HEADINGS
        assert "figures" in RELEVANT_HEADINGS  # Recently added

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

    def test_academic_structure_detection_constants(self):
        """Test that academic structure detection uses reasonable thresholds."""
        # This tests the fixed behavior from the bug fix
        processor = ContentProcessor()

        # Single academic heading should be sufficient
        abstract_only = [HeadingInfo("Abstract", 2, 0, 10, "", is_relevant=True)]
        assert processor._has_academic_structure(abstract_only) == True

        # Two relevant headings should be sufficient
        two_relevant = [
            HeadingInfo("Results", 2, 0, 10, "", is_relevant=True),
            HeadingInfo("Methods", 2, 20, 30, "", is_relevant=True),
        ]
        assert processor._has_academic_structure(two_relevant) == True


class TestDuplicateRemovalRegressions:
    """Regression tests for the duplicate removal bug that removed academic content."""

    @pytest.fixture
    def processor(self):
        return ContentProcessor()

    def test_nejm_structure_preserves_academic_content(self, processor):
        """
        Regression test for NEJM-style documents with paywall H1s after academic content.
        The bug: duplicate removal kept paywall content and removed academic content.
        """
        markdown = """
Skip to main content
Sign In | Create Account

# Mutations in Sarcomere Protein Genes as a Cause of Dilated Cardiomyopathy

## Abstract
### Background
The molecular basis of idiopathic dilated cardiomyopathy is largely unknown.
### Methods
Clinical evaluations were performed in 21 kindreds with familial dilated cardiomyopathy.
### Results
A genetic locus for mutations associated with dilated cardiomyopathy was identified.
### Conclusions
Mutations in sarcomere protein genes account for approximately 10 percent of cases.

## Methods
### Clinical Evaluation
Written informed consent was obtained from all participants.

## Results
Clinical investigations demonstrated autosomal dominant transmission.

## Discussion
These findings provide insights into the pathogenetic mechanisms of this disease.

## References
Reference 1: Previous study
Reference 2: Another study

# Sign In
{* signInForm *}
Remember Me
Forgot your password?

# Sign In
{* signInForm *} 
Don't have an account?

# Forgot Password
Enter the email address associated with your account.

# Create New Password
We've sent an email with instructions.
        """

        result = processor.refine_article(markdown)

        # Must preserve academic content
        assert "Mutations in Sarcomere Protein Genes" in result
        assert "The molecular basis of idiopathic dilated cardiomyopathy" in result
        assert "Clinical evaluations were performed" in result
        assert "A genetic locus for mutations" in result
        assert "autosomal dominant transmission" in result
        assert "pathogenetic mechanisms" in result

        # Must remove paywall content
        assert "signInForm" not in result
        assert "Forgot your password" not in result
        assert "Create New Password" not in result

        # Should be substantial content (not empty or tiny)
        # Note: References section is correctly removed, so expect ~750-800 chars
        assert len(result) > 700

    def test_legitimate_duplicate_removal_still_works(self, processor):
        """Test that legitimate duplicate removal scenarios still work correctly."""
        markdown = """
# Article Title

Some introductory content that might be duplicated.

## Abstract
This is the first abstract.

## Introduction  
First introduction content.

# Article Title

## Abstract  
This is the second abstract with the real content.

## Introduction
Second introduction with the actual content.

## Methods
Research methodology.

## Results
Study findings.
        """

        result = processor.refine_article(markdown)

        # Should remove the first duplicate and keep the second
        assert "second abstract with the real content" in result
        assert "Second introduction with the actual content" in result
        assert "Research methodology" in result
        assert "Study findings" in result

        # Should not contain the first duplicate content
        first_abstract_count = result.count("This is the first abstract")
        assert first_abstract_count == 0

    def test_mixed_h1_headings_academic_vs_chrome(self, processor):
        """Test documents with mix of academic and chrome/paywall H1 headings."""
        markdown = """
# Research Study: Novel Therapeutic Approaches

## Abstract
This study examines innovative treatment methods.

## Methods
Standard protocols were followed.

## Results
Significant improvements were observed.

# Subscribe Now
Get full access to our journal.

# Login Required
Please sign in to continue reading.

# Contact Us
For technical support contact our team.
        """

        result = processor.refine_article(markdown)

        # Academic content should be preserved
        assert "Research Study: Novel Therapeutic Approaches" in result
        assert "innovative treatment methods" in result
        assert "Standard protocols were followed" in result
        assert "Significant improvements were observed" in result

        # Chrome content should be removed
        assert "Subscribe Now" not in result
        assert "Login Required" not in result
        assert "Contact Us" not in result

    def test_paywall_h1_classification_prevents_bad_cutoffs(self, processor):
        """Test that paywall H1s are properly classified and don't become cutoff points."""
        # Create a document where the processor should identify paywall headings
        markdown = """
Navigation | Home | About

# Important Research Findings

## Abstract
This research reveals important discoveries.

## Methods
Experimental design and procedures.

# Sign In
Username: ___________
Password: ___________
        """

        # Extract and classify headings to verify paywall detection
        headings = processor._extract_and_classify_headings(markdown)

        # Find the "Sign In" heading
        signin_heading = next(
            (h for h in headings if "sign in" in h.text.lower()), None
        )
        assert signin_heading is not None, "Sign In heading should be found"

        # The Sign In heading should be classified as paywall or irrelevant
        assert signin_heading.is_paywall or signin_heading.is_irrelevant, (
            "Sign In heading should be classified as paywall or irrelevant"
        )

        # The full processing should preserve academic content
        result = processor.refine_article(markdown)
        assert "Important Research Findings" in result
        assert "important discoveries" in result
        assert "Experimental design" in result

    def test_empty_result_prevention(self, processor):
        """Test that duplicate removal never results in completely empty content when academic content exists."""
        markdown = """
# Paper Title

Introductory text.

## Abstract
Valid academic abstract content.

## Methods
Valid methodology.

# Some Chrome Heading
Navigation or paywall content.

# Another Chrome Heading  
More non-academic content.
        """

        result = processor.refine_article(markdown)

        # Should never return empty result when academic content exists
        assert len(result.strip()) > 0
        assert "academic abstract content" in result or "Valid methodology" in result

    def test_multiple_abstracts_edge_case(self, processor):
        """Test edge case with multiple abstract sections."""
        markdown = """
# Article Title

## Abstract
First abstract - might be incomplete.

Some content.

# Article Title (Repeated)

## Abstract
Second abstract - complete version with full content.

## Methods
Full methodology section.

## Results  
Complete results section.
        """

        result = processor.refine_article(markdown)

        # Should prefer the more complete version
        assert "complete version with full content" in result
        assert "Full methodology section" in result
        assert "Complete results section" in result
