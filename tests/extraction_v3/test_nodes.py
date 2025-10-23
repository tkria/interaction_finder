import asyncio
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pydantic_graph import GraphRunContext

from interaction_finder.extraction_graph_v3.cache import (
    SemanticCacheManager,
    compute_assessment_cache_key,
    compute_extraction_cache_key,
)
from interaction_finder.extraction_graph_v3.deps import ExtractionDepsV3
from interaction_finder.extraction_graph_v3.nodes import (
    AssessIndividually,
    EvaluatePairs,
    ExtractEntities,
    GeneratePairCandidates,
)
from interaction_finder.extraction_graph_v3.state import ExtractionStateV3
from interaction_finder.extraction_graph_v3.models import PairCandidate
from interaction_finder.extraction_graph_v2.models import (
    EntityWithQuotes,
    EntityPairOut,
    IndividualAssessment,
)
from interaction_finder.models import Term
from interaction_finder.resources import ResourcePool


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.task.get_kind_names.return_value = ["gene", "disease"]
    config.task.relation = "gene-disease interaction"
    config.task.context = "biomedical research"
    config.agents.get.return_value = MagicMock(llm="openai:gpt-4o-mini")
    return config


@pytest.fixture
def mock_page_fetcher():
    return MagicMock()


@pytest.fixture
def sample_resource_pool():
    pool = ResourcePool()
    pool.add(
        url="https://example.com/doc1",
        title="BRCA1 in breast cancer",
        document_text=(
            "BRCA1 is a tumor suppressor gene associated with breast cancer. "
            "Mutations in BRCA1 significantly increase the risk of developing breast cancer."
        ),
    )
    pool.add(
        url="https://example.com/doc2",
        title="BRCA1 in ovarian cancer",
        document_text=(
            "BRCA1 mutations are also linked to ovarian cancer. "
            "Women with BRCA1 mutations have elevated ovarian cancer risk."
        ),
    )
    return pool


@pytest.fixture
def deps_v3(mock_config, mock_page_fetcher):
    return ExtractionDepsV3(
        model="openai:gpt-4o-mini",
        config=mock_config,
        page_fetcher=mock_page_fetcher,
        target_term=Term(name="BRCA1", kind="gene"),
        extraction_parallelism=2,
        semantic_cache_enabled=False,
    )


@pytest.fixture
def state_v3(sample_resource_pool):
    state = ExtractionStateV3()
    state.resource_pool = sample_resource_pool
    return state


def make_entity(resource, phrase, name, kind):
    quote = resource.quote(phrase)
    return EntityWithQuotes(
        name=name,
        kind=kind,
        aliases=[name],
        quotes=[quote],
        confidence=0.9,
    )


# ---------------------------------------------------------------------------
# ExtractEntities tests
# ---------------------------------------------------------------------------


class TestExtractEntities:
    @pytest.mark.asyncio
    async def test_extract_entities_without_cache(self, state_v3, deps_v3):
        resources = list(state_v3.resource_pool.resources)
        mock_entities = [
            [make_entity(resources[0], "BRCA1 is a tumor suppressor gene associated with breast cancer.", "BRCA1", "gene")],
            [make_entity(resources[1], "BRCA1 mutations are also linked to ovarian cancer.", "BRCA1", "gene")],
        ]

        mock_extract = AsyncMock(side_effect=mock_entities)

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.extract_from_resource",
            mock_extract,
        ):
            ctx = GraphRunContext(state=state_v3, deps=deps_v3)
            await ExtractEntities().run(ctx)

        assert "BRCA1" in state_v3.entities_found
        assert mock_extract.await_count == len(resources)
        assert state_v3.metrics.entities_extraction_calls == len(resources)
        assert state_v3.metrics.cache_hits_extraction == 0

    @pytest.mark.asyncio
    async def test_extract_entities_with_cache_miss(
        self, state_v3, deps_v3, tmp_path
    ):
        deps_v3.semantic_cache_enabled = True
        state_v3.cache = SemanticCacheManager(tmp_path / "cache")

        resources = list(state_v3.resource_pool.resources)
        mock_entities = [
            [make_entity(resources[0], "BRCA1 is a tumor suppressor gene associated with breast cancer.", "BRCA1", "gene")]
        ]

        mock_extract = AsyncMock(side_effect=mock_entities + [[]])

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.extract_from_resource",
            mock_extract,
        ):
            ctx = GraphRunContext(state=state_v3, deps=deps_v3)
            await ExtractEntities().run(ctx)

        assert state_v3.metrics.cache_misses_extraction > 0
        assert state_v3.metrics.cache_hits_extraction == 0

    @pytest.mark.asyncio
    async def test_extract_entities_with_cache_hit(
        self, state_v3, deps_v3, tmp_path
    ):
        deps_v3.semantic_cache_enabled = True
        state_v3.cache = SemanticCacheManager(tmp_path / "cache")

        resource = state_v3.resource_pool.resources[0]
        cached_entities = [
            make_entity(
                resource,
                "BRCA1 is a tumor suppressor gene associated with breast cancer.",
                "BRCA1",
                "gene",
            )
        ]

        cache_key = compute_extraction_cache_key(
            full_doc_text=resource.text,
            entity_kinds=deps_v3.get_entity_kinds(),
            model_version=str(deps_v3.model),
            prompt_version="v3_2025-10",
        )
        await state_v3.cache.set_extraction(cache_key, cached_entities)

        mock_extract = AsyncMock(return_value=[])

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.extract_from_resource",
            mock_extract,
        ):
            ctx = GraphRunContext(state=state_v3, deps=deps_v3)
            await ExtractEntities().run(ctx)

        assert state_v3.metrics.cache_hits_extraction > 0
        assert mock_extract.await_count == len(state_v3.resource_pool.resources) - 1

    @pytest.mark.asyncio
    async def test_extract_entities_error_handling(self, state_v3, deps_v3):
        mock_extract = AsyncMock(side_effect=Exception("LLM error"))

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.extract_from_resource",
            mock_extract,
        ):
            ctx = GraphRunContext(state=state_v3, deps=deps_v3)
            await ExtractEntities().run(ctx)

        assert state_v3.metrics.entities_extraction_calls == len(
            state_v3.resource_pool.resources
        )

    @pytest.mark.asyncio
    async def test_extract_entities_parallelism(self, deps_v3):
        pool = ResourcePool()
        for idx in range(4):
            pool.add(
                url=f"https://example.com/doc{idx}",
                title=f"Doc {idx}",
                document_text=f"Document {idx} discusses BRCA1.",
            )

        state = ExtractionStateV3()
        state.resource_pool = pool

        async def fake_extract(resource, *args, **kwargs):
            await asyncio.sleep(0.05)
            phrase = resource.text.split(".")[0] + "."
            return [
                make_entity(
                    resource,
                    phrase,
                    "BRCA1",
                    "gene",
                )
            ]

        mock_extract = AsyncMock(side_effect=fake_extract)

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.extract_from_resource",
            mock_extract,
        ):
            ctx = GraphRunContext(state=state, deps=deps_v3)
            await ExtractEntities().run(ctx)

        assert mock_extract.await_count == 4
        assert "BRCA1" in state.entities_found


# ---------------------------------------------------------------------------
# AssessIndividually tests
# ---------------------------------------------------------------------------


class TestAssessIndividually:
    @pytest.fixture
    def state_with_entities(self, sample_resource_pool):
        state = ExtractionStateV3()
        state.resource_pool = sample_resource_pool
        resources = sample_resource_pool.resources
        state.entities_found["BRCA1"] = make_entity(
            resources[0],
            "BRCA1 is a tumor suppressor gene associated with breast cancer.",
            "BRCA1",
            "gene",
        )
        state.entities_found["breast cancer"] = make_entity(
            resources[0],
            "breast cancer",
            "breast cancer",
            "disease",
        )
        return state

    @pytest.mark.asyncio
    async def test_assess_without_cache(
        self, state_with_entities, deps_v3, sample_resource_pool
    ):
        resources = sample_resource_pool.resources
        assessments = [
            IndividualAssessment(
                entity=state_with_entities.entities_found["BRCA1"],
                relationship_potential="high",
                related_entities=["breast cancer"],
                evidence_quotes=[
                    make_entity(
                        resources[0],
                        "BRCA1 is a tumor suppressor gene associated with breast cancer.",
                        "BRCA1",
                        "gene",
                    ).quotes[0]
                ],
                reasoning="Strong evidence",
                confidence=0.9,
            ),
            IndividualAssessment(
                entity=state_with_entities.entities_found["breast cancer"],
                relationship_potential="medium",
                related_entities=[],
                evidence_quotes=[],
                reasoning="Background entity",
                confidence=0.7,
            ),
        ]

        mock_analyze = AsyncMock(side_effect=assessments)

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.analyze_entity",
            mock_analyze,
        ):
            ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)
            await AssessIndividually().run(ctx)

        assert len(state_with_entities.individual_assessments) == 2
        assert state_with_entities.metrics.assessment_calls == 2

    @pytest.mark.asyncio
    async def test_assess_with_cache_hit(
        self, state_with_entities, deps_v3, tmp_path
    ):
        deps_v3.semantic_cache_enabled = True
        state_with_entities.cache = SemanticCacheManager(tmp_path / "cache")

        assessment = IndividualAssessment(
            entity=next(iter(state_with_entities.entities_found.values())),
            relationship_potential="high",
            related_entities=[],
            evidence_quotes=[],
            reasoning="Cached",
            confidence=0.8,
        )

        cache_key = compute_assessment_cache_key(
            entity_name=assessment.entity.name,
            entity_kind=assessment.entity.kind,
            task_context=deps_v3.get_task_context(),
            model_version=str(deps_v3.model),
            prompt_version="v3_2025-10",
        )
        await state_with_entities.cache.set_assessment(cache_key, assessment)

        mock_analyze = AsyncMock(return_value=assessment)

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.analyze_entity",
            mock_analyze,
        ):
            ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)
            await AssessIndividually().run(ctx)

        assert state_with_entities.metrics.cache_hits_assessment > 0
        assert mock_analyze.await_count == len(state_with_entities.entities_found) - 1

    @pytest.mark.asyncio
    async def test_assess_error_handling(self, state_with_entities, deps_v3):
        mock_analyze = AsyncMock(side_effect=Exception("analysis error"))

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.analyze_entity",
            mock_analyze,
        ):
            ctx = GraphRunContext(state=state_with_entities, deps=deps_v3)
            await AssessIndividually().run(ctx)

        assert state_with_entities.metrics.assessment_calls == len(
            state_with_entities.entities_found
        )


# ---------------------------------------------------------------------------
# GeneratePairCandidates tests
# ---------------------------------------------------------------------------


class TestGeneratePairCandidates:
    def _setup_state(self, sample_resource_pool):
        state = ExtractionStateV3()
        state.resource_pool = sample_resource_pool
        resources = sample_resource_pool.resources

        gene = make_entity(
            resources[0],
            "BRCA1 is a tumor suppressor gene associated with breast cancer.",
            "BRCA1",
            "gene",
        )
        disease = make_entity(
            resources[0], "breast cancer", "breast cancer", "disease"
        )
        state.entities_found = {"BRCA1": gene, "breast cancer": disease}
        assessment = IndividualAssessment(
            entity=gene,
            relationship_potential="high",
            related_entities=["breast cancer"],
            evidence_quotes=gene.quotes,
            reasoning="Strong relationship",
            confidence=0.9,
        )
        state.individual_assessments = {"BRCA1": assessment}
        return state

    def test_generate_candidates(self, deps_v3, sample_resource_pool):
        state = self._setup_state(sample_resource_pool)
        ctx = GraphRunContext(state=state, deps=deps_v3)

        node = GeneratePairCandidates()
        asyncio.run(node.run(ctx))

        assert len(state.pair_candidates) >= 1
        candidate = next(iter(state.pair_candidates.values()))
        assert candidate.entity_a.name == "BRCA1"
        assert candidate.entity_b.name == "breast cancer"


# ---------------------------------------------------------------------------
# EvaluatePairs tests
# ---------------------------------------------------------------------------


class TestEvaluatePairs:
    @pytest.fixture
    def state_with_candidates(self, sample_resource_pool):
        state = ExtractionStateV3()
        state.resource_pool = sample_resource_pool
        resources = sample_resource_pool.resources

        gene = make_entity(
            resources[0],
            "BRCA1 is a tumor suppressor gene associated with breast cancer.",
            "BRCA1",
            "gene",
        )
        disease = make_entity(
            resources[0], "breast cancer", "breast cancer", "disease"
        )

        state.entities_found = {"BRCA1": gene, "breast cancer": disease}

        candidate = PairCandidate(
            entity_a=gene,
            entity_b=disease,
            co_occurrence_count=1,
            shared_resources=[resources[0].id.id],
            generation_strategy="same_chunk",
        )
        state.pair_candidates = {("BRCA1", "breast cancer"): candidate}
        return state

    @pytest.mark.asyncio
    async def test_evaluate_pairs_accept(
        self, state_with_candidates, deps_v3, sample_resource_pool
    ):
        resources = sample_resource_pool.resources
        pair = EntityPairOut(
            entity_a=state_with_candidates.entities_found["BRCA1"],
            entity_b=state_with_candidates.entities_found["breast cancer"],
            relationship="gene-disease interaction",
            confidence="high",
            evidence_quotes=[
                make_entity(
                    resources[0],
                    "BRCA1 is a tumor suppressor gene associated with breast cancer.",
                    "BRCA1",
                    "gene",
                ).quotes[0]
            ],
            reasoning="Accepted",
        )

        mock_evaluate = AsyncMock(return_value=pair)

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.evaluate_pair",
            mock_evaluate,
        ):
            ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)
            await EvaluatePairs().run(ctx)

        assert len(state_with_candidates.final_pairs) == 1
        assert state_with_candidates.metrics.pairs_accepted == 1

    @pytest.mark.asyncio
    async def test_evaluate_pairs_reject(self, state_with_candidates, deps_v3):
        mock_evaluate = AsyncMock(return_value=None)

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.evaluate_pair",
            mock_evaluate,
        ):
            ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)
            await EvaluatePairs().run(ctx)

        assert len(state_with_candidates.final_pairs) == 0
        assert state_with_candidates.metrics.pairs_rejected >= 0

    @pytest.mark.asyncio
    async def test_evaluate_pairs_error(self, state_with_candidates, deps_v3):
        mock_evaluate = AsyncMock(side_effect=Exception("evaluation error"))

        with patch(
            "interaction_finder.extraction_graph_v3.nodes.evaluate_pair",
            mock_evaluate,
        ):
            ctx = GraphRunContext(state=state_with_candidates, deps=deps_v3)
            await EvaluatePairs().run(ctx)

        assert state_with_candidates.metrics.pair_evaluation_calls == 1
