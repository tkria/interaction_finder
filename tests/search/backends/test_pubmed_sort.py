"""
Tests for PubMed backend sort parameter support.

Verifies that sort_by filter is correctly mapped to PubMed API parameters,
and that backward compatibility is maintained when sort_by is not specified.
"""

from interaction_finder.search.backends.pubmed import PubMedBackend
from interaction_finder.search.base import SearchQuery


class TestPubMedSort:
    """Test sort parameter mapping in PubMed backend."""

    def setup_method(self):
        """Create PubMed backend instance."""
        # Minimal config without API key for testing parameter building
        self.backend = PubMedBackend(config={})

    def test_sort_relevance(self):
        """Test relevance sort parameter mapping."""
        query = SearchQuery(
            query="BRCA1", max_results=100, filters={"sort_by": "relevance"}
        )
        params = self.backend._build_search_params(query)
        # Check that sort parameter is set to relevance
        assert params["sort"] == "relevance"
        # Check that no sort_order is set (default ascending)
        assert "sort_order" not in params

    def test_sort_date_ascending(self):
        """Test date sort (oldest first) parameter mapping."""
        query = SearchQuery(
            query="diabetes", max_results=50, filters={"sort_by": "date"}
        )
        params = self.backend._build_search_params(query)
        # Check that sort parameter is set to pub_date
        assert params["sort"] == "pub_date"
        # Check that no sort_order is set (default ascending = oldest first)
        assert "sort_order" not in params

    def test_sort_date_descending(self):
        """Test date sort (newest first) parameter mapping."""
        query = SearchQuery(
            query="COVID-19", max_results=100, filters={"sort_by": "date_desc"}
        )
        params = self.backend._build_search_params(query)
        # Check that sort parameter is set to pub_date
        assert params["sort"] == "pub_date"
        # Check that sort_order is set to desc (newest first)
        assert params["sort_order"] == "desc"

    def test_no_sort_parameter(self):
        """Test backward compatibility when sort_by not specified."""
        query = SearchQuery(query="cancer", max_results=100)
        params = self.backend._build_search_params(query)
        # Check that no sort parameter is added (PubMed default applies)
        assert "sort" not in params
        assert "sort_order" not in params

    def test_sort_with_other_filters(self):
        """Test sort parameter works alongside other filters."""
        query = SearchQuery(
            query="heart disease",
            max_results=100,
            filters={
                "sort_by": "relevance",
                "date_range": {"start": "2020", "end": "2023"},
                "publication_type": ["Clinical Trial"],
            },
        )
        params = self.backend._build_search_params(query)
        # Check that sort parameter is set
        assert params["sort"] == "relevance"
        # Check that term includes date filter (filters are processed)
        assert "Date - Publication" in params["term"]
        assert "Clinical Trial" in params["term"]

    def test_sort_invalid_value_defaults_to_relevance(self):
        """Test graceful handling of invalid sort_by values."""
        query = SearchQuery(
            query="test", max_results=10, filters={"sort_by": "invalid_sort_value"}
        )
        params = self.backend._build_search_params(query)
        # Check that invalid value defaults to relevance
        assert params["sort"] == "relevance"
        # Check that no sort_order is set for invalid values
        assert "sort_order" not in params

    def test_empty_filters_dict(self):
        """Test backward compatibility with empty filters dict."""
        query = SearchQuery(query="test", max_results=10, filters={})
        params = self.backend._build_search_params(query)
        # Check that no sort parameters are added
        assert "sort" not in params
        assert "sort_order" not in params

    def test_filters_without_sort_by(self):
        """Test backward compatibility with filters but no sort_by."""
        query = SearchQuery(query="test", max_results=10, filters={"journal": "Nature"})
        params = self.backend._build_search_params(query)
        # Check that no sort parameters are added
        assert "sort" not in params
        assert "sort_order" not in params
        # Check that other filter is still processed
        assert "Nature" in params["term"]
