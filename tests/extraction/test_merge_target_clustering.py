"""Test that merge targets participate in clustering (regression test for bug)."""

from interaction_finder.extraction.entity_matching import (
    find_consolidation_candidates,
    SpeculatedVariant,
    SPEC_ORIGINAL,
)


def test_merge_targets_participate_in_clustering():
    """Merge targets (parents) should participate in clustering with other entities.

    Regression test for bug where merge targets were excluded from clustering,
    preventing parent entities from forming groups with siblings.

    Example: After auto-merging "pulmonary arterial hypertension" (lowercase) to
    "Pulmonary arterial hypertension" (capitalized), the canonical form should
    still cluster with related entities like "Primary PAH", "Idiopathic PAH", etc.

    Bug history: entity_matching.py line 872 previously added both merge sources
    AND targets to already_handled, preventing targets from clustering. This meant
    canonical entities after auto-merge couldn't form hierarchical groups.
    """
    # Simulate entities after variant extraction
    entities = {
        # This will be auto-merged to capitalized version (fuzzy match)
        "pulmonary arterial hypertension": [
            SpeculatedVariant(
                "pulmonary arterial hypertension", SPEC_ORIGINAL, "original"
            )
        ],
        # Capitalized canonical form - this is the merge target
        "Pulmonary arterial hypertension": [
            SpeculatedVariant(
                "Pulmonary arterial hypertension", SPEC_ORIGINAL, "original"
            )
        ],
        # Related entities that should cluster with PAH
        "Primary pulmonary arterial hypertension": [
            SpeculatedVariant(
                "Primary pulmonary arterial hypertension", SPEC_ORIGINAL, "original"
            )
        ],
        "Idiopathic pulmonary arterial hypertension": [
            SpeculatedVariant(
                "Idiopathic pulmonary arterial hypertension", SPEC_ORIGINAL, "original"
            )
        ],
        "Familial pulmonary arterial hypertension": [
            SpeculatedVariant(
                "Familial pulmonary arterial hypertension", SPEC_ORIGINAL, "original"
            )
        ],
    }

    candidates = find_consolidation_candidates(entities, cluster_threshold=0.50)

    # Check auto-merge happened
    assert len(candidates.auto_merge) >= 1
    lowercase_merged = any(
        source == "pulmonary arterial hypertension"
        for source, target, _ in candidates.auto_merge
    )
    assert lowercase_merged, "Expected lowercase PAH to be auto-merged"

    # The key assertion: merge target should participate in clustering
    # After auto-merge, we should have:
    # - "Pulmonary arterial hypertension" (merge target - should cluster!)
    # - "Primary pulmonary arterial hypertension"
    # - "Idiopathic pulmonary arterial hypertension"
    # - "Familial pulmonary arterial hypertension"
    # These 4 entities should form a cluster group

    # Check that clustering produced groups
    assert len(candidates.agent_review_groups) > 0, (
        "Expected clustering to produce groups, but got none. "
        "This suggests merge targets are still being excluded from clustering."
    )

    # Find the PAH cluster
    pah_groups = [
        group
        for group in candidates.agent_review_groups
        if "Pulmonary arterial hypertension" in group
    ]

    assert len(pah_groups) >= 1, (
        "Expected 'Pulmonary arterial hypertension' (merge target) to appear in a cluster group. "
        "This confirms merge targets now participate in clustering."
    )

    # The PAH group should contain the canonical form + related entities
    pah_group = pah_groups[0]
    assert len(pah_group) >= 3, (
        f"Expected PAH cluster to have 3+ entities, got {len(pah_group)}: {pah_group}. "
        "The merge target should cluster with sibling entities."
    )

    # Verify the canonical form is in the group
    assert "Pulmonary arterial hypertension" in pah_group

    # Verify siblings are in the group
    expected_in_group = {
        "Primary pulmonary arterial hypertension",
        "Idiopathic pulmonary arterial hypertension",
        "Familial pulmonary arterial hypertension",
    }
    actual_siblings = pah_group & expected_in_group
    assert len(actual_siblings) >= 2, (
        f"Expected at least 2 PAH siblings in cluster, got {len(actual_siblings)}"
    )
