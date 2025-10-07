"""
Pydantic-AI agents for the extraction graph pipeline.

This module defines all the AI agents used in the graph, following
the pydantic-graph pattern of typed outputs and dependency injection.
"""

from typing import Type, List
from pydantic import BaseModel
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.result import FinalResult
from pydantic_ai.models import Model

try:
    import logfire
except ImportError:
    # Create a no-op logfire if not available
    class _NoOpLogfire:
        def info(self, *args, **kwargs):
            pass

        def warning(self, *args, **kwargs):
            pass

        def error(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    logfire = _NoOpLogfire()


from .deps import Deps
from .models import (
    RouteDecision,
    EntityListOut,
    EntityAssessmentOut,
    EntityPairOut,
    ValidationOut,
    IndividualEntityContext,
    IndividualEntityAssessment,
    EntityAggregationOut,
)
from .usage_tracker import record_agent_usage


async def run_agent_with_tracking(
    agent: Agent, prompt: str, model_name: str, deps: Deps = None, **kwargs
) -> FinalResult:
    """
    Run an agent and track its token usage with verbose progress updates.

    Args:
        agent: The agent to run
        prompt: Input prompt
        model_name: Name of the model (for usage tracking)
        deps: Dependencies to pass to agent
        **kwargs: Additional arguments for agent.run()

    Returns:
        FinalResult from the agent
    """
    import time

    # Extract agent type for logging
    agent_type = getattr(
        agent, "_name", str(type(agent)).split(".")[-1].replace("'>", "")
    )
    prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt

    logfire.info(f"Starting {agent_type} ({model_name}): {prompt_preview}")
    start_time = time.time()
    # Get progress tracker for verbose updates from global context
    # Note: We can't pass state as a parameter to agent.run(), but we can
    # access it through the deps or other means if needed for progress tracking
    progress_tracker = None
    if deps and deps.verbose:
        # For now, progress tracking will be handled at the node level
        # rather than individual agent calls
        pass

    # Update progress tracker with agent call info
    if progress_tracker:
        # Extract agent type from agent or use generic description
        agent_type = getattr(
            agent, "_name", str(type(agent)).split(".")[-1].replace("'>", "")
        )
        # Truncate long prompts for display
        prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
        progress_tracker.update_stage(
            f"Calling {agent_type} ({model_name}): {prompt_preview}"
        )

    # Run the agent with error handling for verbose mode
    try:
        if deps is not None:
            result = await agent.run(prompt, deps=deps, **kwargs)
        else:
            result = await agent.run(prompt, **kwargs)
    except Exception as e:
        agent_time = time.time() - start_time
        logfire.error(f"{agent_type} failed after {agent_time:.1f}s: {str(e)}")
        # Report error to progress tracker if available
        if progress_tracker:
            progress_tracker.report_error(str(e), f"Agent {agent_type} call failed")
        # Re-raise the exception
        raise

    # Record usage if available
    tokens_used = 0
    if hasattr(result, "usage") and result.usage is not None:
        record_agent_usage(model_name, result.usage)
        if hasattr(result.usage, "total_tokens"):
            tokens_used = result.usage.total_tokens
        elif hasattr(result.usage, "request_tokens") and hasattr(
            result.usage, "response_tokens"
        ):
            tokens_used = result.usage.request_tokens + result.usage.response_tokens
    elif hasattr(result, "_usage") and result._usage is not None:
        record_agent_usage(model_name, result._usage)
        if hasattr(result._usage, "total_tokens"):
            tokens_used = result._usage.total_tokens
        elif hasattr(result._usage, "request_tokens") and hasattr(
            result._usage, "response_tokens"
        ):
            tokens_used = result._usage.request_tokens + result._usage.response_tokens

    # Update token usage in progress tracker
    if progress_tracker and tokens_used > 0:
        progress_tracker.update_tokens(tokens_used)

    # Log successful completion
    agent_time = time.time() - start_time
    logfire.info(
        f"{agent_type} completed in {agent_time:.1f}s, tokens used: {tokens_used}"
    )

    return result


def mk_agent(
    model: str | Model,
    output_type: Type[BaseModel],
    system_prompt: str,
    retries: int = 2,
    output_retries: int = 2,
) -> Agent[Deps, BaseModel]:
    """
    Factory for creating agents with consistent configuration.

    Following the pydantic-graph manual pattern for agent creation.

    Args:
        model: Model identifier (e.g., 'openai:gpt-4o')
        output_type: Pydantic model for agent output
        system_prompt: System prompt for the agent
        retries: Number of retries for provider/tool errors
        output_retries: Number of retries for JSON validation errors

    Returns:
        Configured Agent instance
    """
    return Agent(
        model,
        deps_type=Deps,
        output_type=output_type,
        retries=retries,
        output_retries=output_retries,
        system_prompt=system_prompt,
    )


def create_router_agent(
    model: str | Model, entity_kinds: List[str], task_context: str
) -> Agent[Deps, RouteDecision]:
    """Create the initial routing agent."""

    kinds_str = ", ".join(entity_kinds)
    system_prompt = f"""You are a scientific literature classifier.

Your task is to determine if document content is suitable for extracting {kinds_str} entities and their relationships.

CLASSIFICATION CRITERIA:
- "scientific": Contains peer-reviewed research content relevant to {kinds_str}
- "non-scientific": News articles, opinion pieces, non-research content
- "unclear": Ambiguous content that might contain relevant information

CONTEXT: {task_context}

Provide your reasoning clearly and assign an appropriate confidence score."""

    return mk_agent(model, RouteDecision, system_prompt)


def create_entity_extractor_agent(
    model: str | Model, entity_kinds: List[str], task_context: str
) -> Agent[Deps, EntityListOut]:
    """Create the entity extraction agent."""

    kinds_str = ", ".join(entity_kinds)
    system_prompt = f"""You are an expert entity extractor for scientific literature with resource tracking capabilities.

Your task is to identify and extract ALL {kinds_str} entities from the provided text WITH RESOURCE PROVENANCE.

EXTRACTION GUIDELINES:
1. Extract entities of these types: {kinds_str}
2. For each entity, provide:
   - Clean, standardized name
   - Confidence score (0-1)
   - Context where found
   - Any synonyms/alternative names
3. Only extract entities that are clearly identifiable
4. Maintain high precision - better to miss an entity than extract incorrectly

RESOURCE TRACKING REQUIREMENTS:
- When available, you will receive document resources with unique Resource IDs
- For each extracted entity, provide supporting ResourceQuotes if possible
- ResourceQuotes should contain the exact text spans where entities are mentioned
- This enables complete provenance tracking from extraction back to source documents

CONTEXT: {task_context}

Return entities with complete attribution to their source documents, including ResourceQuotes when available."""

    agent = mk_agent(model, EntityListOut, system_prompt)

    @agent.output_validator
    def validate_entities(out: EntityListOut) -> EntityListOut:
        """Validate extracted entities for completeness."""
        if not out.entities and not out.processing_notes:
            raise ModelRetry(
                "No entities found and no processing notes provided. Explain why no entities were found."
            )

        for entity in out.entities:
            if not entity.name or not entity.kind:
                raise ModelRetry("All entities must have both name and kind specified.")

            if entity.kind not in out.kinds_searched:
                raise ModelRetry(
                    f"Entity kind '{entity.kind}' not in searched kinds {out.kinds_searched}"
                )

        return out

    return agent


def create_assessment_agent(
    model: str | Model, entity_kinds: List[str], relation_type: str, task_context: str
) -> Agent[Deps, EntityAssessmentOut]:
    """Create the entity relationship assessment agent."""

    kinds_str = ", ".join(entity_kinds)
    system_prompt = f"""You are an expert in analyzing {kinds_str} relationships in scientific literature with resource tracking capabilities.

Your task is to assess whether two entities have a {relation_type} relationship based on the provided evidence WITH COMPLETE PROVENANCE.

ASSESSMENT GUIDELINES:
1. Analyze the relationship between the two provided entities
2. Look for evidence of {relation_type} in the text
3. Assign confidence: "high", "moderate", or "low"
4. Extract specific text evidence supporting the relationship
5. Explain your reasoning clearly

RESOURCE TRACKING REQUIREMENTS:
- When available, provide ResourceQuotes for all evidence claims
- ResourceQuotes should contain the exact text spans that support the relationship
- Include resource IDs to enable tracing evidence back to source documents
- This ensures that every relationship claim can be verified against original sources

CONTEXT: {task_context}

Be precise and conservative - only claim relationships that have clear textual support, and provide ResourceQuotes whenever possible."""

    agent = mk_agent(model, EntityAssessmentOut, system_prompt)

    @agent.output_validator
    def validate_assessment(out: EntityAssessmentOut) -> EntityAssessmentOut:
        """Validate relationship assessment for completeness."""
        if not out.evidence:
            raise ModelRetry(
                "Must provide at least one piece of evidence for the relationship assessment."
            )

        if not out.reasoning:
            raise ModelRetry("Must provide reasoning for the relationship assessment.")

        if out.relationship_type != relation_type:
            raise ModelRetry(
                f"Relationship type must be '{relation_type}', got '{out.relationship_type}'"
            )

        return out

    return agent


def create_pair_generator_agent(
    model: str | Model, entity_kinds: List[str], relation_type: str, task_context: str
) -> Agent[Deps, EntityPairOut]:
    """Create the entity pair generation agent."""

    kinds_str = ", ".join(entity_kinds)
    system_prompt = f"""You are an expert in generating validated {kinds_str} entity pairs with comprehensive resource tracking.

Your task is to create high-quality entity pairs with {relation_type} relationships based on individual assessments WITH COMPLETE PROVENANCE.

PAIR GENERATION GUIDELINES:
1. Combine individual entity assessments into coherent pairs
2. CRITICAL: Only create pairs between DIFFERENT entity types (e.g., gene-disease, not disease-disease)
3. For gene-disease tasks: genes must pair with diseases, never gene-gene or disease-disease
4. Calculate overall confidence score (0-1)
5. Compile all supporting evidence
6. Ensure complete source document attribution
7. Add assessment details for transparency

RESOURCE TRACKING REQUIREMENTS:
- Consolidate ResourceQuotes from all contributing assessments
- Ensure every evidence claim is backed by specific ResourceQuotes
- Maintain traceability from final pairs back to original document sources
- Include resource IDs for all supporting evidence

CONTEXT: {task_context}

Focus on creating pairs with strong evidence and clear relationships, ensuring complete resource provenance."""

    agent = mk_agent(model, EntityPairOut, system_prompt)

    @agent.output_validator
    def validate_pair(out: EntityPairOut) -> EntityPairOut:
        """Validate entity pair for completeness and quality."""
        if not out.evidence:
            raise ModelRetry("Entity pairs must have supporting evidence.")

        if not out.source_documents:
            raise ModelRetry("Entity pairs must specify source documents.")

        if out.confidence <= 0 or out.confidence > 1:
            raise ModelRetry("Confidence must be between 0 and 1 (exclusive of 0).")

        # Validate that pairs are between different entity types
        if out.entity_a.kind == out.entity_b.kind:
            raise ModelRetry(
                f"Pairs must be between different entity types, not {out.entity_a.kind}-{out.entity_b.kind}"
            )

        return out

    return agent


def create_validation_agent(
    model: str | Model, entity_kinds: List[str], relation_type: str, task_context: str
) -> Agent[Deps, ValidationOut]:
    """Create the quality validation agent."""

    kinds_str = ", ".join(entity_kinds)
    system_prompt = f"""You are a quality control expert for {kinds_str} entity pair extraction.

Your task is to validate entity pairs and provide feedback for improvement.

VALIDATION CRITERIA:
1. Evidence quality: Is the evidence specific and relevant?
2. Relationship clarity: Is the {relation_type} relationship well-supported?
3. Entity accuracy: Are the entities correctly identified and typed?
4. Completeness: Is all necessary information provided?

DECISIONS:
- "approve": High quality, ready for final output
- "revise": Good but needs improvement, provide specific feedback
- "reject": Poor quality, fundamental issues

CONTEXT: {task_context}

Provide actionable feedback for any issues found."""

    agent = mk_agent(model, ValidationOut, system_prompt)

    @agent.output_validator
    def validate_validation(out: ValidationOut) -> ValidationOut:
        """Ensure validation output is actionable."""
        if out.verdict == "revise" and not out.suggestions:
            raise ModelRetry(
                "When requesting revision, must provide specific suggestions."
            )

        if out.verdict == "reject" and not out.issues_found:
            raise ModelRetry("When rejecting, must specify issues found.")

        return out

    return agent


def create_context_extractor_agent(
    model: str | Model, entity_kinds: List[str], task_context: str
) -> Agent[Deps, IndividualEntityContext]:
    """Create agent for extracting context around a single entity."""

    kinds_str = ", ".join(entity_kinds)
    system_prompt = f"""You are an expert at extracting detailed context for individual entities from scientific text.

Your task is to analyze a single entity and extract comprehensive context information.

CONTEXT EXTRACTION GUIDELINES:
1. Find ALL direct mentions of the specific entity in the text
2. Extract claims made about this entity (what it does, properties, relationships)
3. Gather evidence supporting claims about the entity
4. Identify relevant text chunks that contain information about the entity
5. Assess confidence in the extracted context

ENTITY TYPES: {kinds_str}
CONTEXT: {task_context}

Focus on this ONE entity only. Extract thorough context while maintaining high precision."""

    agent = mk_agent(model, IndividualEntityContext, system_prompt)

    @agent.output_validator
    def validate_context(out: IndividualEntityContext) -> IndividualEntityContext:
        if not out.mentions and not out.claims and not out.evidence:
            raise ModelRetry(
                "Must find at least one mention, claim, or piece of evidence about the entity"
            )

        if out.context_confidence <= 0:
            raise ModelRetry("Context confidence must be greater than 0")

        return out

    return agent


def create_individual_assessor_agent(
    model: str | Model, entity_kinds: List[str], relation_type: str, task_context: str
) -> Agent[Deps, IndividualEntityAssessment]:
    """Create agent for assessing individual entity's relationship potential."""

    kinds_str = ", ".join(entity_kinds)
    system_prompt = f"""You are an expert at assessing individual entities for {relation_type} relationship potential.

Your task is to evaluate a single entity's context and determine its potential for relationships.

ASSESSMENT GUIDELINES:
1. Analyze the entity's extracted context (mentions, claims, evidence)
2. Assess relationship potential: "high", "moderate", "low", or "none"
3. Identify potential target entities mentioned in the context
4. Provide detailed reasoning for the assessment
5. Assign confidence score based on evidence quality

ENTITY TYPES: {kinds_str}
RELATIONSHIP TYPE: {relation_type}
CONTEXT: {task_context}

Focus on identifying relationship signals and potential partners for this ONE entity."""

    agent = mk_agent(model, IndividualEntityAssessment, system_prompt)

    @agent.output_validator
    def validate_assessment(
        out: IndividualEntityAssessment,
    ) -> IndividualEntityAssessment:
        if not out.reasoning:
            raise ModelRetry("Must provide detailed reasoning for the assessment")

        if out.relationship_potential == "high" and not out.target_entity_hints:
            raise ModelRetry(
                "High potential relationships must identify specific target entity hints"
            )

        if out.confidence <= 0:
            raise ModelRetry("Assessment confidence must be greater than 0")

        return out

    return agent


def create_aggregation_agent(
    model: str | Model, entity_kinds: List[str], relation_type: str, task_context: str
) -> Agent[Deps, EntityAggregationOut]:
    """Create agent for aggregating individual assessments into entity pairs."""

    kinds_str = ", ".join(entity_kinds)
    system_prompt = f"""You are an expert at combining individual entity assessments into validated {relation_type} pairs.

Your task is to analyze multiple individual entity assessments and create coherent pairs.

AGGREGATION GUIDELINES:
1. Review all individual entity assessments and their contexts
2. Match entities based on:
   - Mutual target entity hints
   - Complementary relationship potential
   - Overlapping evidence and contexts
3. CRITICAL: Only create pairs between DIFFERENT entity types (e.g., gene-disease, not disease-disease)
4. For gene-disease tasks: genes must pair with diseases, never gene-gene or disease-disease
5. Create pairs with confidence levels (high/moderate)
6. CRITICAL: For each EntityPairOut, populate source_documents with ALL URLs from both entities
7. Include evidence from both entity assessments in the evidence field
8. Identify entities without clear pair matches
9. Provide detailed reasoning for aggregation decisions

SOURCE ATTRIBUTION REQUIREMENTS:
- Every EntityPairOut MUST have source_documents populated with actual URLs
- source_documents should include source_url from both entity_a and entity_b
- Never leave source_documents empty - this breaks provenance tracking

ENTITY TYPES: {kinds_str}
RELATIONSHIP TYPE: {relation_type}
CONTEXT: {task_context}

Focus on creating well-evidenced pairs while maintaining conservative standards."""

    agent = mk_agent(model, EntityAggregationOut, system_prompt)

    @agent.output_validator
    def validate_aggregation(out: EntityAggregationOut) -> EntityAggregationOut:
        if not out.aggregation_reasoning:
            raise ModelRetry("Must provide reasoning for aggregation decisions")

        if out.aggregation_confidence <= 0:
            raise ModelRetry("Aggregation confidence must be greater than 0")

        if out.total_individuals_processed <= 0:
            raise ModelRetry("Must specify how many individuals were processed")

        # Validate that all pairs have source documents
        for pair in out.high_confidence_pairs + out.moderate_confidence_pairs:
            if not pair.source_documents:
                raise ModelRetry(
                    "All EntityPairOut objects must have source_documents populated with actual URLs"
                )

            # Validate that pairs are between different entity types
            if pair.entity_a.kind == pair.entity_b.kind:
                raise ModelRetry(
                    f"Pairs must be between different entity types, not {pair.entity_a.kind}-{pair.entity_b.kind}"
                )

        return out

    return agent


def create_extraction_agents(config, model: str | None = None) -> dict:
    """
    Create all agents needed for the extraction graph.

    Args:
        config: IfetcherConfig instance
        model: Optional model override

    Returns:
        Dictionary of configured agents
    """
    # Get configuration
    entity_kinds = config.task.get_kind_names()
    relation_type = config.task.relation
    task_context = config.task.context

    # Determine model
    if model is None:
        default_agent = config.agents.get("_", config.AgentSpec())
        model = default_agent.llm or "openai:gpt-4o"

    return {
        "router": create_router_agent(model, entity_kinds, task_context),
        "extractor": create_entity_extractor_agent(model, entity_kinds, task_context),
        "assessor": create_assessment_agent(
            model, entity_kinds, relation_type, task_context
        ),
        "pair_generator": create_pair_generator_agent(
            model, entity_kinds, relation_type, task_context
        ),
        "validator": create_validation_agent(
            model, entity_kinds, relation_type, task_context
        ),
        # New individual processing agents
        "context_extractor": create_context_extractor_agent(
            model, entity_kinds, task_context
        ),
        "individual_assessor": create_individual_assessor_agent(
            model, entity_kinds, relation_type, task_context
        ),
        "aggregator": create_aggregation_agent(
            model, entity_kinds, relation_type, task_context
        ),
    }
