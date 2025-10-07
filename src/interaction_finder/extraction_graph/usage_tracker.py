"""
Token usage tracking for pydantic-ai agent calls.

This module provides functionality to track and aggregate token usage
across all AI agent calls during extraction processing.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class UsageMetrics:
    """Token usage metrics for a single model."""

    request_tokens: int = 0
    response_tokens: int = 0
    cached_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    accepted_prediction_tokens: int = 0
    rejected_prediction_tokens: int = 0

    def add(self, other: "UsageMetrics") -> None:
        """Add another UsageMetrics to this one."""
        self.request_tokens += other.request_tokens
        self.response_tokens += other.response_tokens
        self.cached_tokens += other.cached_tokens
        self.total_tokens += other.total_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.accepted_prediction_tokens += other.accepted_prediction_tokens
        self.rejected_prediction_tokens += other.rejected_prediction_tokens

    def calculate_total(self) -> None:
        """Calculate total tokens if not already set."""
        if self.total_tokens == 0:
            self.total_tokens = (
                self.request_tokens
                + self.response_tokens
                + self.cached_tokens
                + self.reasoning_tokens
                + self.accepted_prediction_tokens
                + self.rejected_prediction_tokens
            )

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary format for JSON serialization."""
        self.calculate_total()
        return {
            "request_tokens": self.request_tokens,
            "response_tokens": self.response_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "accepted_prediction_tokens": self.accepted_prediction_tokens,
            "rejected_prediction_tokens": self.rejected_prediction_tokens,
        }


class UsageTracker:
    """
    Tracks token usage across all agent calls during extraction.

    Supports both single-model flat structure and per-model mapping
    as specified in the output format requirements.
    """

    def __init__(self):
        """Initialize empty usage tracker."""
        self.model_usage: Dict[str, UsageMetrics] = defaultdict(UsageMetrics)
        self._call_count = 0

    def record_usage(
        self,
        model_name: str,
        request_tokens: int = 0,
        response_tokens: int = 0,
        cached_tokens: int = 0,
        total_tokens: int = 0,
        reasoning_tokens: int = 0,
        accepted_prediction_tokens: int = 0,
        rejected_prediction_tokens: int = 0,
        **kwargs,  # Ignore additional fields
    ) -> None:
        """
        Record token usage for a model.

        Args:
            model_name: Name of the model used
            request_tokens: Tokens in request
            response_tokens: Tokens in response
            cached_tokens: Cached tokens used
            total_tokens: Total tokens (auto-calculated if 0)
            reasoning_tokens: Reasoning tokens used
            accepted_prediction_tokens: Accepted prediction tokens
            rejected_prediction_tokens: Rejected prediction tokens
        """
        # Sanitize model name (remove provider prefix)
        if ":" in model_name:
            model_name = model_name.split(":", 1)[1]

        # Create usage metrics
        usage = UsageMetrics(
            request_tokens=request_tokens,
            response_tokens=response_tokens,
            cached_tokens=cached_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
            accepted_prediction_tokens=accepted_prediction_tokens,
            rejected_prediction_tokens=rejected_prediction_tokens,
        )

        # Add to model totals
        self.model_usage[model_name].add(usage)
        self._call_count += 1

    def record_pydantic_ai_usage(self, model_name: str, usage_data: Any) -> None:
        """
        Record usage from pydantic-ai FinalResult or similar object.

        Args:
            model_name: Name of the model used
            usage_data: Usage data from pydantic-ai (could be various formats)
        """
        try:
            # Handle different pydantic-ai usage data formats
            if hasattr(usage_data, "model_dump"):
                # Pydantic model
                usage_dict = usage_data.model_dump()
            elif hasattr(usage_data, "__dict__"):
                # Object with attributes
                usage_dict = usage_data.__dict__
            elif isinstance(usage_data, dict):
                # Already a dictionary
                usage_dict = usage_data
            else:
                # Fallback - try to extract what we can
                usage_dict = {}

            # Extract standard fields with various naming conventions
            request_tokens = (
                usage_dict.get("request_tokens", 0)
                or usage_dict.get("input_tokens", 0)
                or usage_dict.get("prompt_tokens", 0)
                or 0
            )

            response_tokens = (
                usage_dict.get("response_tokens", 0)
                or usage_dict.get("output_tokens", 0)
                or usage_dict.get("completion_tokens", 0)
                or 0
            )

            cached_tokens = usage_dict.get("cached_tokens", 0)
            total_tokens = (
                usage_dict.get("total_tokens", 0) or usage_dict.get("total", 0) or 0
            )
            reasoning_tokens = usage_dict.get("reasoning_tokens", 0)
            accepted_prediction_tokens = usage_dict.get("accepted_prediction_tokens", 0)
            rejected_prediction_tokens = usage_dict.get("rejected_prediction_tokens", 0)

            # Record the usage
            self.record_usage(
                model_name=model_name,
                request_tokens=request_tokens,
                response_tokens=response_tokens,
                cached_tokens=cached_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens,
                accepted_prediction_tokens=accepted_prediction_tokens,
                rejected_prediction_tokens=rejected_prediction_tokens,
            )

        except Exception as e:
            # If usage tracking fails, continue silently
            # but record that we made an attempt
            self._call_count += 1

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Get usage summary in the required format.

        Returns:
            Dictionary formatted per specification:
            - Single model: flat structure
            - Multiple models: per-model mapping
        """
        if not self.model_usage:
            # No usage recorded
            return {}

        if len(self.model_usage) == 1:
            # Single model - return flat structure
            model_name = next(iter(self.model_usage.keys()))
            return self.model_usage[model_name].to_dict()
        else:
            # Multiple models - return per-model mapping
            return {
                model_name: metrics.to_dict()
                for model_name, metrics in self.model_usage.items()
            }

    def get_total_usage(self) -> UsageMetrics:
        """
        Get total usage across all models.

        Returns:
            UsageMetrics with totals across all models
        """
        total = UsageMetrics()
        for metrics in self.model_usage.values():
            total.add(metrics)
        return total

    def reset(self) -> None:
        """Reset usage tracking."""
        self.model_usage.clear()
        self._call_count = 0

    @property
    def call_count(self) -> int:
        """Number of agent calls recorded."""
        return self._call_count

    @property
    def model_names(self) -> list[str]:
        """List of model names that have usage recorded."""
        return list(self.model_usage.keys())


# Global usage tracker instance
_global_tracker: Optional[UsageTracker] = None


def get_global_tracker() -> UsageTracker:
    """Get the global usage tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = UsageTracker()
    return _global_tracker


def reset_global_tracker() -> None:
    """Reset the global usage tracker."""
    global _global_tracker
    _global_tracker = None


def record_agent_usage(model_name: str, usage_data: Any) -> None:
    """
    Record usage from an agent call to the global tracker.

    Args:
        model_name: Name of the model used
        usage_data: Usage data from pydantic-ai
    """
    tracker = get_global_tracker()
    tracker.record_pydantic_ai_usage(model_name, usage_data)


def get_usage_summary() -> Dict[str, Any]:
    """Get usage summary from global tracker."""
    tracker = get_global_tracker()
    return tracker.get_usage_summary()
