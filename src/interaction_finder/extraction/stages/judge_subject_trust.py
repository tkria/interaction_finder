"""Subject-side taxonomic trust gate (gate_review v3).

For each judged pair, ask one direct question of its *subject-side* entity --
"is this the right kind of subject for the topic?" -- and record the verdict on
the judgment. This MARKS pairs; it never removes them. Downstream consumers
(e.g. the report's strict-by-default filter) decide what to show.

The subject side is the entity whose kind equals ``state.subject_kind`` -- a
single value declared once for the run, not inferred per pair. Pairs with no
subject-kind entity are left untouched (``subject_trust`` stays None). The judge
is run once per unique subject entity and the verdict fanned out to every pair
that shares it.

If ``state.subject_kind`` is unset, the stage is a no-op -- the gate simply did
not run for this extraction.
"""

import asyncio
from collections import defaultdict

from interaction_finder.agent_config import AGENT_CALL_ERRORS
from interaction_finder.agent_utils import rename_agent
from interaction_finder.extraction.deps import Deps
from interaction_finder.extraction.judge_subject_trust import (
    get_subject_trust_agent,
)
from interaction_finder.extraction.models import SimpleEntity, SubjectTrustVerdict
from interaction_finder.extraction.state import State
from interaction_finder.usage import record_usage
from interaction_finder.logging import logfire


def _subject_entity(judgment, subject_kind: str) -> SimpleEntity | None:
    """Return the pair's subject-side entity (first whose kind matches), or None.

    Cross-kind domains (disease-gene, celltype-marker, ligand-receptor) have
    exactly one subject-kind entity per pair. For a self-pair (both sides the
    subject kind) the first entity is judged.
    """
    for entity in (judgment.entity1, judgment.entity2):
        if entity.kind == subject_kind:
            return entity
    return None


async def judge_subject_trust(state: State, deps: Deps) -> bool:
    """Mark every pair with a subject-side taxonomic trust verdict (v3)."""
    with logfire.span("judge_subject_trust"):
        subject_kind = state.subject_kind
        anchor = state.subject_anchor
        if not subject_kind or not anchor:
            deps.logger.info(
                "judge_subject_trust: subject_kind/subject_anchor unset; "
                "skipping subject gate."
            )
            return True

        # Group pairs by their unique subject-side entity name so the judge is
        # called once per subject and fanned out to every pair that shares it.
        pairs_by_subject: dict[str, list] = defaultdict(list)
        for judgment in state.pair_judgments.values():
            subject = _subject_entity(judgment, subject_kind)
            if subject is not None:
                pairs_by_subject[subject.name].append(judgment)

        if not pairs_by_subject:
            deps.logger.info(
                f"judge_subject_trust: no pairs with a {subject_kind!r} entity; "
                "nothing to judge."
            )
            return True

        if deps.progress:
            deps.progress["Unique pairs"].activate()
            deps.progress.set_status("Subject-trust gate")

        async def _judge_subject(subject_name: str, judgments: list) -> None:
            verdict = await _verdict_for(
                subject_name, subject_kind, state, deps
            )
            if verdict is None:
                return
            for judgment in judgments:
                judgment.subject_trust = verdict

        tasks = [
            _judge_subject(name, judgments)
            for name, judgments in pairs_by_subject.items()
        ]
        for coro in asyncio.as_completed(tasks):
            await coro
        return True


async def _verdict_for(
    candidate: str, subject_kind: str, state: State, deps: Deps
) -> SubjectTrustVerdict | None:
    """Judge one candidate against the run's anchor subject; None on failure.

    Subject-explicit: the anchor (``state.subject_anchor``, the topic's subject
    entity) is passed directly, so the judge compares two same-kind entities and
    never has to infer the subject from the topic phrasing.
    """
    prompt = (
        f"Subject: {state.subject_anchor}\n"
        f"Kind: {subject_kind}\n"
        f"Candidate: {candidate}"
    )
    agent = get_subject_trust_agent(deps.config)
    try:
        with rename_agent(agent, name=f"judge_subject_trust: {candidate}"):
            async with deps.agent_semaphore:
                if deps.progress:
                    deps.progress["Unique pairs"].work()
                result = await agent.run(prompt, deps=deps)
        record_usage(deps.usage, "subject_trust_judge", agent, result)
        return SubjectTrustVerdict(
            belongs=result.output.belongs,
            reasoning=result.output.reasoning,
            subject_name=candidate,
            subject_kind=subject_kind,
        )
    except AGENT_CALL_ERRORS as e:
        deps.logger.error(
            f"Subject-trust judgment failed for {candidate!r}: "
            f"{type(e).__name__}: {e}"
        )
        return None
