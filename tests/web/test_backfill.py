"""Tests for reconstructing an event stream from a saved checkpoint."""

from __future__ import annotations

from interaction_finder.checkpoint import (
    ExtractionStageData,
    KeywordsStageData,
    PipelineCheckpoint,
    SearchStageData,
)
from interaction_finder.extraction.models import (
    ExtractionMetadata,
    PairJudgment,
    PairSpread,
    SimpleEntity,
)
from interaction_finder.resources import ResourcePool
from interaction_finder.search.models import SearchResult
from interaction_finder.web.backfill import checkpoint_to_events


def _evidence(level: int):
    from interaction_finder.extraction.models import EvidenceQuality

    return EvidenceQuality(
        directness="explicit",
        source_type="primary",
        specificity="associative",
        language="definitive",
        overall=level,
    )


def _judgment(e1: str, e2: str, *, accepted: bool, evidence: int) -> PairJudgment:
    return PairJudgment(
        topic_relevance=3,
        entity1=SimpleEntity(name=e1, kind="gene", aliases=[e1]),
        entity2=SimpleEntity(name=e2, kind="disease", aliases=[e2]),
        relationship="associated_with",
        spread=PairSpread(),
        accepted=accepted,
        evidence=_evidence(evidence),
        decision_confidence=0.9,
        reasoning="test",
    )


def _extraction(judgments) -> ExtractionStageData:
    return ExtractionStageData(
        target_entity_types=["gene", "disease"],
        permitted_pairs={"gene": ["disease"]},
        judgments=judgments,
        metadata=ExtractionMetadata(
            topic="t",
            resource_count=1,
            total_entities_found=2,
            entities_after_validation=2,
            entities_merged=0,
            merge_cache_hits=0,
            merge_cache_misses=0,
            proximal_sets_found=1,
            total_pairs_found=len(judgments),
            pairs_accepted=sum(j.accepted for j in judgments),
            pairs_rejected=sum(not j.accepted for j in judgments),
            quotes_validated=1,
            quotes_failed=0,
        ),
    )


def test_empty_checkpoint_yields_no_events():
    checkpoint = PipelineCheckpoint(topic="t", resources=ResourcePool())
    assert checkpoint_to_events(checkpoint) == []


def test_keywords_backfill_matches_live_shape():
    checkpoint = PipelineCheckpoint(
        topic="t",
        resources=ResourcePool(),
        keywords=KeywordsStageData(
            terms=["BMPR2", "endothelin"],
            scores=[0.9, 0.7],
            total_documents_processed=3,
            rounds_completed=1,
            coverage_assessment="Comprehensive coverage of the research domain achieved",
            resource_urls=["http://example.com/1"],
        ),
    )
    events = checkpoint_to_events(checkpoint)
    # Boundary then the scored event, matching the live keywords.scored payload.
    assert events[0].scope == "meta.cleared"
    scored = events[1]
    assert scored.scope == "keywords.scored"
    assert scored.data["terms"] == [
        {"term": "BMPR2", "score": 0.9},
        {"term": "endothelin", "score": 0.7},
    ]


def test_search_backfill_resolves_selected_urls_to_titles():
    results = [
        SearchResult(
            title="Genetic basis of PAH",
            url="http://a",
            snippet="A review of PAH genetics",
            relevance=0.9,
        ),
        SearchResult(title="BMPR2 signalling", url="http://b", snippet=None),
        SearchResult(title="Unselected review", url="http://c", snippet=None),
    ]
    checkpoint = PipelineCheckpoint(
        topic="t",
        resources=ResourcePool(),
        search=SearchStageData(
            results=results,
            queries=["BMPR2 PAH"],
            query_results={"BMPR2 PAH": ["http://a", "http://b"]},
            keyphrases=["BMPR2"],
            rounds_completed=1,
        ),
    )
    events = checkpoint_to_events(checkpoint)
    selected = [e for e in events if e.scope == "search.selected"]
    assert len(selected) == 1
    assert selected[0].data["query"] == "BMPR2 PAH"
    assert selected[0].data["picked"] == [
        {
            "title": "Genetic basis of PAH",
            "url": "http://a",
            "snippet": "A review of PAH genetics",
            "relevance": 0.9,
        },
        {
            "title": "BMPR2 signalling",
            "url": "http://b",
            "snippet": None,
            "relevance": None,
        },
    ]


def test_extraction_backfill_one_event_per_judgment():
    checkpoint = PipelineCheckpoint(
        topic="t",
        resources=ResourcePool(),
        extraction=_extraction(
            [
                _judgment("TP53", "cancer", accepted=True, evidence=8),
                _judgment("EGFR", "lung disease", accepted=False, evidence=3),
            ]
        ),
    )
    events = checkpoint_to_events(checkpoint)
    judged = [e for e in events if e.scope == "extraction.pair_judged"]
    assert len(judged) == 2
    assert judged[0].data == {
        "entity1": "TP53",
        "entity2": "cancer",
        "relationship": "associated_with",
        "accepted": True,
        "evidence": 8,
    }
    assert judged[1].data["accepted"] is False


def test_full_checkpoint_orders_stages_with_boundaries():
    checkpoint = PipelineCheckpoint(
        topic="t",
        resources=ResourcePool(),
        keywords=KeywordsStageData(
            terms=["x"],
            scores=[0.5],
            total_documents_processed=1,
            rounds_completed=1,
            coverage_assessment="Sufficient coverage of the relevant research literature obtained",
            resource_urls=["http://example.com/1"],
        ),
        search=SearchStageData(
            results=[SearchResult(title="T", url="http://a", snippet=None)],
            queries=["q"],
            query_results={"q": ["http://a"]},
            keyphrases=["x"],
            rounds_completed=1,
        ),
        extraction=_extraction([_judgment("A", "B", accepted=True, evidence=5)]),
    )
    scopes = [e.scope for e in checkpoint_to_events(checkpoint)]
    # Pipeline order, each stage preceded by a boundary marker.
    assert scopes == [
        "meta.cleared",
        "keywords.scored",
        "meta.cleared",
        "search.selected",
        "meta.cleared",
        "extraction.pair_judged",
    ]
