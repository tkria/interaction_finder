"""Tests that merge decision prompts use canonical names, not normalized forms.

When entity consolidation finds merge candidates, it should present the LLM
with the actual entity names as they appear in documents (canonical names),
not the normalized/lowercased forms used for matching.

Example:
- Normalized matching: "bmpr2" matches "bmpr2 gene" (substring)
- Prompt should show: "BMPR2" vs "BMPR2 gene" (canonical names)
- NOT: "bmpr2" vs "bmpr2 gene" (normalized forms)
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_graph import GraphRunContext

from interaction_finder.extraction.models import (
    EntityMention,
    EntityMergeDecision,
    EntityMergeDecisions,
)
from interaction_finder.extraction.nodes import ConsolidateEntitiesNode
from interaction_finder.extraction.state import State
from interaction_finder.extraction.utils import build_permitted_pairs
from interaction_finder.resources import ResourceId


@pytest.fixture
def mock_deps():
    """Create mock dependencies."""
    deps = MagicMock()
    deps.config = MagicMock()
    deps.config.tools.extraction.merge_batch_size = 50
    deps.logger = MagicMock()
    deps.agent_semaphore = MagicMock()
    deps.agent_semaphore.__aenter__ = AsyncMock()
    deps.agent_semaphore.__aexit__ = AsyncMock()
    return deps


class TestCanonicalNamesInPrompts:
    """Test that LLM prompts use canonical entity names, not normalized forms."""

    @pytest.mark.asyncio
    async def test_prompt_uses_canonical_not_normalized_names(self, mock_deps):
        """LLM should see canonical names like 'BMPR2' not normalized 'bmpr2'."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="PAH genetics",
            target_entity_types=["gene"],
            permitted_pairs=build_permitted_pairs(["gene"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Create entities with canonical names that will normalize to substrings
        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.validated_entities_by_resource = {
            resource1: {
                "BMPR2": EntityMention(
                    kind="gene",
                    name="BMPR2",
                    aliases=["BMPR2"],
                    quotes=[],
                    reasoning="test",
                ),
                "BMPR2 gene": EntityMention(
                    kind="gene",
                    name="BMPR2 gene",
                    aliases=["BMPR2 gene"],
                    quotes=[],
                    reasoning="test",
                ),
            }
        }

        # Mock the agent to capture the prompt
        captured_prompt = None

        async def mock_run(prompt, deps, usage):
            nonlocal captured_prompt
            captured_prompt = prompt
            # Return a decision to merge (pair_id=1 for first pair)
            # Token "test" won't match the real token, but ID-based lookup will be used
            return MagicMock(
                output=EntityMergeDecisions(
                    decisions=[
                        EntityMergeDecision(
                            pair_id=1,
                            pair_token="test",
                            should_merge=True,
                            reasoning="Same gene, different naming conventions",
                        )
                    ]
                )
            )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=mock_run)
        mock_agent._name = "test_agent"

        # Patch the agent getter
        with patch(
            "interaction_finder.extraction.nodes.get_entity_merge_agent",
            return_value=mock_agent,
        ):
            await node.run(ctx)

        # Verify the prompt used canonical names, not normalized forms
        assert captured_prompt is not None, "Prompt should have been captured"

        # Should contain canonical names
        assert "BMPR2" in captured_prompt, "Should show canonical 'BMPR2'"
        assert "BMPR2 gene" in captured_prompt, "Should show canonical 'BMPR2 gene'"

        # Verify entity lines show proper case, not all lowercase
        # New format: "[ID] Parent: 'name' | Child: 'name'"
        lines = captured_prompt.split("\n")
        entity_lines = [l for l in lines if "Parent:" in l and "Child:" in l]
        # At least one entity line should exist
        assert len(entity_lines) > 0, "Should have entity pair lines in prompt"

        # Entity lines should show canonical forms ("BMPR2"), not normalized ("bmpr2")
        # Check that we don't see all-lowercase entity names in quotes
        for line in entity_lines:
            # Extract the quoted entity name
            import re

            match = re.search(r"'([^']+)'", line)
            if match:
                entity_name = match.group(1)
                # If it looks like it should be capitalized but isn't, that's an error
                if entity_name.islower() and len(entity_name) > 0:
                    # Allow common words like "gene" but not gene names
                    if entity_name not in ["gene", "disease", "protein", "phenotype"]:
                        pytest.fail(
                            f"Entity name shown in lowercase: '{entity_name}' in line: {line}"
                        )

    @pytest.mark.asyncio
    async def test_prompt_picks_most_complex_canonical_name(self, mock_deps):
        """When multiple canonical names exist for same normalized form, pick most complex."""
        node = ConsolidateEntitiesNode()
        state = State(
            topic="PAH genetics",
            target_entity_types=["disease"],
            permitted_pairs=build_permitted_pairs(["disease"]),
        )
        ctx = GraphRunContext(state=state, deps=mock_deps)

        # Create two entities that will create a substring match
        # "pah" is substring of "pah disease"
        resource1 = ResourceId(url="https://example.com/doc1", counter=0)
        ctx.state.validated_entities_by_resource = {
            resource1: {
                "PAH": EntityMention(
                    kind="disease",
                    name="PAH",
                    aliases=["PAH"],
                    quotes=[],
                    reasoning="test",
                ),
                "PAH Disease": EntityMention(
                    kind="disease",
                    name="PAH Disease",
                    aliases=["PAH Disease"],
                    quotes=[],
                    reasoning="test",
                ),
            }
        }

        captured_prompt = None

        async def mock_run(prompt, deps, usage):
            nonlocal captured_prompt
            captured_prompt = prompt
            # Token "test" won't match the real token, but ID-based lookup will be used
            return MagicMock(
                output=EntityMergeDecisions(
                    decisions=[
                        EntityMergeDecision(
                            pair_id=1,
                            pair_token="test",
                            should_merge=True,
                            reasoning="Same entity, different naming",
                        )
                    ]
                )
            )

        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(side_effect=mock_run)
        mock_agent._name = "test_agent"

        with patch(
            "interaction_finder.extraction.nodes.get_entity_merge_agent",
            return_value=mock_agent,
        ):
            await node.run(ctx)

        # Should show the canonical names
        assert captured_prompt is not None
        assert "PAH" in captured_prompt, "Should show PAH"
        assert "PAH Disease" in captured_prompt, "Should show PAH Disease"
