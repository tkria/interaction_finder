"""
Quote validation logic and validators.
"""

import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING
from pydantic_ai import ModelRetry
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from datetime import datetime

from .alignment import SequenceAligner, ErrorType
from .corrections import QuoteCorrector
from .error_messages import build_retry_message
from ...fetcher.utils import url_to_hash_base36

if TYPE_CHECKING:
    from ...resources import Resource, ResourceQuote
    from ..models import QuoteErrorRecord

logger = logging.getLogger(__name__)


def _get_cache_filename(url: str) -> str:
    """
    Compute cache filename for a URL.

    Checks for actual cache files with collision resolution suffixes
    and returns the filename if found, or the expected base filename.

    Args:
        url: Resource URL

    Returns:
        Cache filename (e.g. "abc123de.md" or "abc123de_1.md")
    """
    base_hash = url_to_hash_base36(url)

    # Check for cache files with collision resolution
    # Try both 'cache' (default) and '.cache' (alternative) directories
    for cache_dirname in ["cache", ".cache"]:
        cache_dir = Path(cache_dirname)
        for probe in range(10):  # COLLISION_PROBE_LIMIT from cache.py
            hash_str = base_hash if probe == 0 else f"{base_hash}_{probe}"
            cache_path = cache_dir / f"{hash_str}.md"
            if cache_path.exists():
                return f"{hash_str}.md"

    # No existing file found, return expected base filename
    return f"{base_hash}.md"


class QuoteValidator:
    """Handles quote validation with alignment-based auto-correction and error logging."""

    def __init__(
        self,
        auto_accept_threshold: float = 85.0,
        default_similarity_threshold: float = 1.0,
    ):
        """
        Initialize quote validator.

        Args:
            auto_accept_threshold: Threshold for auto-accepting corrections (%)
            default_similarity_threshold: Default similarity threshold to use when
                constructing ResourceQuote instances.
        """
        self.corrector = QuoteCorrector(auto_accept_threshold)
        self.aligner = SequenceAligner()
        self.console = Console()
        self.default_similarity_threshold = default_similarity_threshold

    def validate_and_correct_quote(
        self,
        entity_name: str,
        entity_kind: str,
        quote_text: str,
        resource: "Resource",
        quote_error_log: List["QuoteErrorRecord"],
        current_retry: int = 0,
        similarity_threshold: Optional[float] = None,
    ) -> Optional["ResourceQuote"]:
        """
        Validate a quote and attempt auto-correction if validation fails.

        Args:
            entity_name: Name of the entity
            entity_kind: Kind of entity (gene, disease, etc.)
            quote_text: The quote text to validate
            resource: Resource to validate against
            quote_error_log: List to append error records to
            current_retry: Current retry attempt number
            similarity_threshold: Minimum similarity required when matching quote
                text against the resource. Defaults to validator setting.

        Returns:
            ResourceQuote if successful, None if validation fails
        """
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else self.default_similarity_threshold
        )

        # First try to create ResourceQuote directly
        try:
            return resource.quote(quote_text, similarity_threshold=threshold)
        except Exception as original_error:
            # Quote validation failed - try auto-correction
            return self._handle_quote_failure(
                entity_name=entity_name,
                entity_kind=entity_kind,
                quote_text=quote_text,
                resource=resource,
                original_error=original_error,
                quote_error_log=quote_error_log,
                current_retry=current_retry,
                similarity_threshold=threshold,
            )

    def _handle_quote_failure(
        self,
        entity_name: str,
        entity_kind: str,
        quote_text: str,
        resource: "Resource",
        original_error: Exception,
        quote_error_log: List["QuoteErrorRecord"],
        current_retry: int,
        similarity_threshold: float,
    ) -> Optional["ResourceQuote"]:
        """Handle quote validation failure with correction attempts."""
        from ..models import QuoteErrorRecord

        # Generate correction suggestions using alignment analysis
        suggestions = self.corrector.generate_suggestions(
            quote_text, resource, self.aligner
        )

        # Check if we can auto-accept using enhanced alignment-based confidence
        if suggestions and self._should_auto_accept_with_alignment(
            suggestions, quote_text, resource
        ):
            best_suggestion = self.corrector.get_best_suggestion(suggestions)

            try:
                # Try to use the auto-corrected quote
                corrected_quote = resource.quote(
                    best_suggestion, similarity_threshold=similarity_threshold
                )

                # Get alignment details for logging
                alignment = self.aligner.align_quote_to_resource(
                    best_suggestion, resource
                )

                # Log the auto-correction with alignment info
                self._log_auto_correction(
                    entity_name,
                    quote_text,
                    best_suggestion,
                    suggestions[0][1],
                    alignment,
                )

                # Record as resolved error with enhanced details
                error_record = QuoteErrorRecord(
                    entity_name=entity_name,
                    entity_kind=entity_kind,
                    original_quote=quote_text,
                    error_type=self._classify_error_type(
                        quote_text, resource, suggestions
                    ),
                    suggested_corrections=[s[0] for s in suggestions[:3]],
                    matched_percentage=suggestions[0][1],
                    final_accepted_quote=best_suggestion,
                    retry_attempt=current_retry,
                    retry_message=f"Auto-corrected with high alignment confidence ({alignment.similarity_ratio:.1%})",
                    resolved=True,
                )
                quote_error_log.append(error_record)

                return corrected_quote

            except Exception:
                # Auto-correction also failed, fall through to manual retry
                logger.warning(
                    f"Auto-correction failed for {entity_name}: {best_suggestion}"
                )

        # Auto-correction not possible or failed - prepare for LLM retry
        self._log_quote_failure(entity_name, quote_text, suggestions, resource)

        # Build retry message
        retry_message = build_retry_message(
            entity_name=entity_name,
            entity_kind=entity_kind,
            failed_quote=quote_text,
            resource=resource,
            suggestions=suggestions,
        )

        # Record the error
        error_record = QuoteErrorRecord(
            entity_name=entity_name,
            entity_kind=entity_kind,
            original_quote=quote_text,
            error_type=self._classify_error_type(quote_text, resource, suggestions),
            suggested_corrections=[s[0] for s in suggestions[:3]],
            matched_percentage=suggestions[0][1] if suggestions else None,
            final_accepted_quote=None,
            retry_attempt=current_retry,
            retry_message=retry_message,
            resolved=False,
        )
        quote_error_log.append(error_record)

        # Raise ModelRetry with the constructed message
        raise ModelRetry(retry_message)

    def _classify_error_type(
        self, quote_text: str, resource: "Resource", suggestions: List[tuple]
    ) -> str:
        """Classify the type of quote error using alignment analysis."""
        # Use alignment analysis for precise error classification
        alignment = self.aligner.align_quote_to_resource(quote_text, resource)

        # Map alignment error types to string format for compatibility
        error_type_mapping = {
            ErrorType.SPLIT_QUOTE: "split_quote",
            ErrorType.WORD_SUBSTITUTION: "word_substitution",
            ErrorType.INSERTION: "insertion",
            ErrorType.DELETION: "deletion",
            ErrorType.PARAPHRASE: "paraphrased",
            ErrorType.REORDERING: "reordering",
            ErrorType.NOT_FOUND: "not_found",
        }

        return error_type_mapping.get(alignment.error_type, "not_found")

    def _should_auto_accept_with_alignment(
        self, suggestions, original_quote: str, resource
    ) -> bool:
        """Enhanced auto-acceptance logic using alignment quality metrics."""
        if not suggestions:
            return False

        best_suggestion, confidence = suggestions[0]

        # Use standard confidence threshold as baseline
        if not self.corrector.should_auto_accept(suggestions):
            return False

        # Additional alignment-based validation
        try:
            # Verify the correction is actually valid by checking alignment
            corrected_alignment = self.aligner.align_quote_to_resource(
                best_suggestion, resource
            )

            # Require high alignment quality for auto-acceptance
            alignment_requirements = (
                corrected_alignment.similarity_ratio > 0.9  # Very high similarity
                and corrected_alignment.coverage_ratio > 0.85  # Good coverage
                and corrected_alignment.contiguous  # Must be contiguous
                and corrected_alignment.error_type
                != ErrorType.SPLIT_QUOTE  # No split quotes
            )

            return alignment_requirements

        except Exception:
            # If alignment analysis fails, fall back to confidence only
            return confidence >= self.corrector.auto_accept_threshold

    def _log_auto_correction(
        self,
        entity_name: str,
        original_quote: str,
        corrected_quote: str,
        confidence: float,
        alignment=None,
    ) -> None:
        """Log successful auto-correction with alignment details."""
        alignment_info = ""
        if alignment:
            alignment_info = (
                f"[blue]Alignment:[/blue] {alignment.similarity_ratio:.1%} similarity, "
                f"{alignment.coverage_ratio:.1%} coverage, "
                f"{'contiguous' if alignment.contiguous else 'fragmented'}\n"
            )

        self.console.print(
            Panel(
                f"[green]✅ AUTO-CORRECTED[/green] {entity_name}\n"
                f"[yellow]Original:[/yellow] {original_quote[:100]}...\n"
                f"[green]Corrected:[/green] {corrected_quote[:100]}...\n"
                f"[blue]Confidence:[/blue] {confidence:.1f}%\n"
                f"{alignment_info}",
                title="Quote Auto-Correction",
                border_style="green",
            )
        )

    def _log_quote_failure(
        self,
        entity_name: str,
        quote_text: str,
        suggestions: List[tuple],
        resource: "Resource",
    ) -> None:
        """Log quote validation failure with rich formatting."""
        if not suggestions:
            suggestion_text = "[red]No corrections available[/red]"
        else:
            best_suggestion, confidence = suggestions[0]
            suggestion_text = f"[yellow]Best suggestion ({confidence:.1f}%):[/yellow]\n{best_suggestion[:100]}..."

        # Get cache filename for manual inspection
        cache_filename = _get_cache_filename(resource.id.url)

        self.console.print(
            Panel(
                f"[red]❌ QUOTE FAILED[/red] {entity_name}\n"
                f"[yellow]Failed quote:[/yellow] {quote_text[:100]}...\n"
                f"{suggestion_text}\n"
                f"[blue]Source:[/blue] {resource.id.url}\n"
                f"[blue]Cache file:[/blue] {cache_filename}",
                title="Quote Validation Error",
                border_style="red",
            )
        )
