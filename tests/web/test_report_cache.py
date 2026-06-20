"""Tests for the report cache key/path."""

from __future__ import annotations

from interaction_finder.checkpoint import (
    ExtractionStageData,
    KeywordsStageData,
    PipelineCheckpoint,
)
from interaction_finder.extraction.models import (
    EvidenceQuality,
    ExtractionMetadata,
    PairJudgment,
    PairSpread,
    SimpleEntity,
)
from interaction_finder.resources import ResourcePool
from interaction_finder.web.report_cache import report_cache_key, report_cache_path


def _evidence(level=7):
    return EvidenceQuality(
        directness="explicit",
        source_type="primary",
        specificity="associative",
        language="definitive",
        overall=level,
    )


def _checkpoint(topic, pairs):
    judgments = [
        PairJudgment(
            topic_relevance=3,
            entity1=SimpleEntity(name=a, kind="gene", aliases=[a]),
            entity2=SimpleEntity(name=b, kind="disease", aliases=[b]),
            relationship="associated_with",
            spread=PairSpread(),
            accepted=acc,
            evidence=_evidence(),
            decision_confidence=0.9,
            reasoning="t",
        )
        for a, b, acc in pairs
    ]
    return PipelineCheckpoint(
        topic=topic,
        resources=ResourcePool(),
        extraction=ExtractionStageData(
            target_entity_types=["gene", "disease"],
            permitted_pairs={"gene": ["disease"]},
            judgments=judgments,
            metadata=ExtractionMetadata(
                topic=topic,
                resource_count=1,
                total_entities_found=2,
                entities_after_validation=2,
                entities_merged=0,
                merge_cache_hits=0,
                merge_cache_misses=0,
                proximal_sets_found=1,
                total_pairs_found=len(judgments),
                pairs_accepted=sum(p[2] for p in pairs),
                pairs_rejected=sum(not p[2] for p in pairs),
                quotes_validated=1,
                quotes_failed=0,
            ),
        ),
    )


def test_same_content_same_key_regardless_of_pair_order():
    a = _checkpoint("PAH", [("TP53", "cancer", True), ("EGFR", "lung", False)])
    b = _checkpoint("PAH", [("EGFR", "lung", False), ("TP53", "cancer", True)])
    assert report_cache_key(a) == report_cache_key(b)


def test_different_topic_changes_key():
    a = _checkpoint("PAH", [("TP53", "cancer", True)])
    b = _checkpoint("ALS", [("TP53", "cancer", True)])
    assert report_cache_key(a) != report_cache_key(b)


def test_different_verdict_changes_key():
    a = _checkpoint("PAH", [("TP53", "cancer", True)])
    b = _checkpoint("PAH", [("TP53", "cancer", False)])
    assert report_cache_key(a) != report_cache_key(b)


def test_cache_path_structure():
    # The cache dir is redirected to a temp dir by the autouse fixture, so this
    # checks structure, not the real platform location.
    ck = _checkpoint("PAH", [("TP53", "cancer", True)])
    path = report_cache_path(ck)
    assert path.suffix == ".html"
    assert path.parent.name == "reports"
    assert path.parent.is_dir()  # created on demand
    assert path.stem == report_cache_key(ck)
