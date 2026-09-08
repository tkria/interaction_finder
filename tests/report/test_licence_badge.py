"""Tests for the licence badge shown with a document's provenance links."""

import pytest

from interaction_finder.report.licence_badge import (
    licence_badge_html,
    licence_url,
    redaction_notice_html,
)


class TestLicenceUrl:
    """Canonical URLs for the licence names OpenAlex reports."""

    @pytest.mark.parametrize(
        "licence,expected",
        [
            ("cc-by", "https://creativecommons.org/licenses/by/4.0/"),
            ("cc-by-sa", "https://creativecommons.org/licenses/by-sa/4.0/"),
            ("cc-by-nc", "https://creativecommons.org/licenses/by-nc/4.0/"),
            ("cc-by-nd", "https://creativecommons.org/licenses/by-nd/4.0/"),
            ("cc-by-nc-nd", "https://creativecommons.org/licenses/by-nc-nd/4.0/"),
            ("cc-by-nc-sa", "https://creativecommons.org/licenses/by-nc-sa/4.0/"),
            ("cc0", "https://creativecommons.org/publicdomain/zero/1.0/"),
            ("public-domain", "https://creativecommons.org/publicdomain/mark/1.0/"),
        ],
    )
    def test_creative_commons_licences_resolve(self, licence, expected):
        assert licence_url(licence) == expected

    def test_case_and_whitespace_are_tolerated(self):
        assert licence_url("  CC-BY  ") == licence_url("cc-by")

    def test_non_creative_commons_licence_has_no_url(self):
        """'other-oa' names no specific licence, so there is nothing to link to."""
        assert licence_url("other-oa") is None

    def test_unknown_licence_has_no_url(self):
        assert licence_url("some-bespoke-terms") is None


class TestLicenceBadgeHtml:
    """The badge markup: a link where one exists, plain text otherwise."""

    def test_badge_links_to_the_licence_deed(self):
        html = licence_badge_html("cc-by")
        assert 'href="https://creativecommons.org/licenses/by/4.0/"' in html
        assert 'class="document-licence"' in html
        assert ">CC BY<" in html

    def test_badge_opens_in_a_new_tab_safely(self):
        html = licence_badge_html("cc-by")
        assert 'target="_blank"' in html
        assert 'rel="noreferrer noopener"' in html

    def test_unlinkable_licence_renders_as_a_span(self):
        html = licence_badge_html("other-oa")
        assert html.startswith("<span")
        assert "href" not in html

    def test_no_licence_renders_nothing(self):
        assert licence_badge_html(None) == ""
        assert licence_badge_html("") == ""

    def test_display_name_is_human_readable(self):
        assert ">CC BY-NC-ND<" in licence_badge_html("cc-by-nc-nd")
        assert ">Public domain<" in licence_badge_html("public-domain")

    def test_licence_name_is_escaped(self):
        """A licence string is external data and must not inject markup."""
        html = licence_badge_html('<script>alert(1)</script>')
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_badge_carries_an_explanatory_tooltip(self):
        assert "title=" in licence_badge_html("cc-by-nc")


class TestRedactionNotice:
    """A reader must be told why a document is shown as extracts."""

    def test_notice_names_a_real_licence_as_the_reason(self):
        html = redaction_notice_html(licence="cc-by-nc-nd")
        assert "not reproduced here" in html
        assert "cc-by-nc-nd" in html

    def test_placeholder_licence_is_not_asserted_to_forbid_sharing(self):
        """`other-oa` names no licence, so its terms are unknown, not hostile."""
        html = redaction_notice_html(licence="other-oa")
        assert "no licence permitting redistribution could be established" in html
        assert "does not permit" not in html

    def test_notice_without_a_licence_says_none_was_established(self):
        html = redaction_notice_html()
        assert "no licence permitting redistribution could be established" in html

    def test_notice_reports_the_withheld_length_when_known(self):
        assert "45,000 characters" in redaction_notice_html(text_length=45000)

    def test_notice_omits_length_when_unknown(self):
        assert "characters" not in redaction_notice_html()

    def test_notice_points_at_the_original(self):
        assert "View original" in redaction_notice_html()

    def test_unrecognised_licence_string_is_never_echoed(self):
        """Only known licence names are named, so external text cannot inject."""
        html = redaction_notice_html(licence="<script>alert(1)</script>")
        assert "script" not in html
        assert "no licence permitting redistribution could be established" in html
