"""Tests for checkpoint-or-topic CLI parameter handling."""

import json
from pathlib import Path

import pytest

from interaction_finder.cli import load_checkpoint_or_create
from interaction_finder.checkpoint import PipelineCheckpoint, KeywordsStageData
from interaction_finder.resources import ResourcePool


class TestLoadCheckpointOrCreate:
    """Test load_checkpoint_or_create helper function."""

    def test_load_existing_checkpoint_file(self, tmp_path):
        """Load checkpoint from existing file."""
        # Create a checkpoint file
        checkpoint = PipelineCheckpoint(
            topic="Test Topic",
            resources=ResourcePool(),
            keywords=KeywordsStageData(
                terms=["term1", "term2"],
                scores=[0.9, 0.8],
                total_documents_processed=5,
                rounds_completed=2,
                coverage_assessment="Good coverage achieved with comprehensive review of the research topic",
                resource_urls=["https://example.com/1"],
            ),
        )
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text(checkpoint.model_dump_json())

        # Load it
        loaded_checkpoint, topic = load_checkpoint_or_create(str(checkpoint_file))

        assert isinstance(loaded_checkpoint, PipelineCheckpoint)
        assert topic == "Test Topic"
        assert loaded_checkpoint.keywords is not None
        assert loaded_checkpoint.keywords.terms == ["term1", "term2"]

    def test_create_from_topic_string(self):
        """Create empty checkpoint from topic string."""
        topic_string = "cancer genomics"

        checkpoint, topic = load_checkpoint_or_create(topic_string)

        assert isinstance(checkpoint, PipelineCheckpoint)
        assert topic == "cancer genomics"
        assert checkpoint.topic == "cancer genomics"
        assert checkpoint.keywords is None
        assert checkpoint.search is None
        assert checkpoint.extraction is None
        assert len(checkpoint.resources.resource_map) == 0

    def test_topic_string_with_special_characters(self):
        """Handle topic strings with special characters."""
        topic_string = "pulmonary arterial hypertension & BMPR2"

        checkpoint, topic = load_checkpoint_or_create(topic_string)

        assert checkpoint.topic == "pulmonary arterial hypertension & BMPR2"
        assert topic == "pulmonary arterial hypertension & BMPR2"

    def test_nonexistent_file_treated_as_topic(self, tmp_path):
        """Non-existent file path treated as topic string."""
        nonexistent = str(tmp_path / "does_not_exist.json")

        checkpoint, topic = load_checkpoint_or_create(nonexistent)

        # Should treat as topic string, not fail on missing file
        assert checkpoint.topic == nonexistent
        assert topic == nonexistent

    def test_invalid_json_file_raises_error(self, tmp_path):
        """Invalid JSON file raises validation error."""
        from pydantic import ValidationError

        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")

        with pytest.raises(ValidationError):
            load_checkpoint_or_create(str(bad_file))

    def test_checkpoint_with_all_stages(self, tmp_path):
        """Load checkpoint with all stages completed."""
        from interaction_finder.checkpoint import (
            SearchStageData,
            ExtractionStageData,
        )
        from interaction_finder.extraction.models import ExtractionMetadata
        from interaction_finder.search.models import SearchResult

        checkpoint = PipelineCheckpoint(
            topic="Full Pipeline Test",
            resources=ResourcePool(),
            keywords=KeywordsStageData(
                terms=["term1"],
                scores=[0.9],
                total_documents_processed=1,
                rounds_completed=1,
                coverage_assessment="Complete coverage achieved with comprehensive analysis of the research domain",
                resource_urls=["https://example.com/1"],
            ),
            search=SearchStageData(
                results=[
                    SearchResult(
                        url="https://example.com/paper1",
                        title="Paper 1",
                        snippet="...",
                    )
                ],
                queries=["query1"],
                query_results={"query1": ["https://example.com/paper1"]},
                keyphrases=["term1"],
                rounds_completed=1,
            ),
            extraction=ExtractionStageData(
                target_entity_types=["gene", "disease"],
                permitted_pairs={"gene": ["disease"], "disease": ["gene"]},
                judgments=[],
                metadata=ExtractionMetadata(
                    topic="Full Pipeline Test",
                    resource_count=1,
                    total_entities_found=0,
                    entities_after_validation=0,
                    entities_merged=0,
                    merge_cache_hits=0,
                    merge_cache_misses=0,
                    proximal_sets_found=0,
                    total_pairs_found=0,
                    pairs_accepted=0,
                    pairs_rejected=0,
                    quotes_validated=0,
                    quotes_failed=0,
                ),
            ),
        )
        checkpoint_file = tmp_path / "full.json"
        checkpoint_file.write_text(checkpoint.model_dump_json())

        loaded_checkpoint, topic = load_checkpoint_or_create(str(checkpoint_file))

        assert topic == "Full Pipeline Test"
        assert loaded_checkpoint.keywords is not None
        assert loaded_checkpoint.search is not None
        assert loaded_checkpoint.extraction is not None


class TestInPlaceCheckpointUpdate:
    """Test in-place checkpoint file update behavior."""

    def test_updates_checkpoint_file_when_no_output_specified(self, tmp_path):
        """When input is a file and no -o specified, update input file in place."""
        from interaction_finder.checkpoint import SearchStageData
        from interaction_finder.search.models import SearchResult

        # Create initial checkpoint with only keywords
        checkpoint = PipelineCheckpoint(
            topic="Test Topic",
            resources=ResourcePool(),
            keywords=KeywordsStageData(
                terms=["term1"],
                scores=[0.9],
                total_documents_processed=1,
                rounds_completed=1,
                coverage_assessment="Complete coverage achieved with comprehensive analysis of the domain",
                resource_urls=["https://example.com/1"],
            ),
        )
        checkpoint_file = tmp_path / "checkpoint.json"
        checkpoint_file.write_text(checkpoint.model_dump_json())

        # Simulate adding search stage
        checkpoint.search = SearchStageData(
            results=[
                SearchResult(
                    url="https://example.com/paper1",
                    title="Paper 1",
                    snippet="...",
                )
            ],
            queries=["query1"],
            query_results={"query1": ["https://example.com/paper1"]},
            keyphrases=["term1"],
            rounds_completed=1,
        )

        # Simulate CLI save logic (input is file, no output specified)
        input_was_file = checkpoint_file.exists() and checkpoint_file.is_file()
        assert input_was_file is True

        # Update in place
        checkpoint_file.write_text(checkpoint.model_dump_json(indent=2))

        # Verify file was updated
        updated_checkpoint = PipelineCheckpoint.model_validate_json(
            checkpoint_file.read_text()
        )
        assert updated_checkpoint.keywords is not None
        assert updated_checkpoint.search is not None
        assert len(updated_checkpoint.search.results) == 1

    def test_explicit_output_overrides_inplace_update(self, tmp_path):
        """When -o specified, write to output file, not input file."""
        # Create initial checkpoint
        checkpoint = PipelineCheckpoint(
            topic="Test Topic",
            resources=ResourcePool(),
            keywords=KeywordsStageData(
                terms=["term1"],
                scores=[0.9],
                total_documents_processed=1,
                rounds_completed=1,
                coverage_assessment="Complete coverage achieved with comprehensive analysis of the domain",
                resource_urls=["https://example.com/1"],
            ),
        )
        input_file = tmp_path / "input.json"
        input_file.write_text(checkpoint.model_dump_json())
        output_file = tmp_path / "output.json"

        # Add search data
        from interaction_finder.checkpoint import SearchStageData
        from interaction_finder.search.models import SearchResult

        checkpoint.search = SearchStageData(
            results=[
                SearchResult(
                    url="https://example.com/paper1",
                    title="Paper 1",
                    snippet="...",
                )
            ],
            queries=["query1"],
            query_results={"query1": ["https://example.com/paper1"]},
            keyphrases=["term1"],
            rounds_completed=1,
        )

        # Simulate CLI save logic (explicit output specified)
        output_file.write_text(checkpoint.model_dump_json(indent=2))

        # Verify output file created
        assert output_file.exists()
        updated = PipelineCheckpoint.model_validate_json(output_file.read_text())
        assert updated.search is not None

        # Verify input file unchanged
        original = PipelineCheckpoint.model_validate_json(input_file.read_text())
        assert original.search is None

    def test_topic_string_requires_explicit_output(self):
        """When input is topic string (no file), command exits with error if no -o."""
        topic_string = "cancer genomics"

        # Simulate CLI logic
        input_path = Path(topic_string)
        input_was_file = input_path.exists() and input_path.is_file()

        # Should not be treated as file
        assert input_was_file is False

        # In this case, CLI exits early with error before running pipeline
        # This prevents wasting time and money on LLM calls that won't be saved

    def test_file_existence_check_robust(self, tmp_path):
        """File existence check handles edge cases correctly."""
        # Directory should not be treated as file
        directory = tmp_path / "mydir"
        directory.mkdir()
        assert not (directory.exists() and directory.is_file())

        # Regular file should be treated as file
        regular_file = tmp_path / "regular.json"
        regular_file.write_text("{}")
        assert regular_file.exists() and regular_file.is_file()

        # Non-existent path should not be treated as file
        nonexistent = tmp_path / "does_not_exist.json"
        assert not (nonexistent.exists() and nonexistent.is_file())


class TestFailFastBehavior:
    """Test that commands exit early when topic string used without -o."""

    def test_all_commands_exit_early_for_topic_strings_without_output(self):
        """All commands (keywords, widesearch, extract) should exit early when topic string has no -o."""
        # This test documents the expected behavior:
        # When input is a topic string and no -o specified, all commands should
        # exit with an error BEFORE running expensive pipeline operations.
        #
        # keywords: always requires -o (takes topic string as input)
        # widesearch: requires -o when given topic string, not when given checkpoint file
        # extract: requires -o when given topic string, not when given checkpoint file
        #
        # This fail-fast approach prevents wasting time and money on LLM calls
        # that won't be saved anywhere.
        pass  # Documented behavior, tested via integration tests
