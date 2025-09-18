"""
Quote validation logic and validators.
"""

import logging
from typing import List, Optional, TYPE_CHECKING
from pydantic_ai import ModelRetry
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from datetime import datetime

from .corrections import QuoteCorrector
from .error_messages import build_retry_message

if TYPE_CHECKING:
    from ...resources import Resource, ResourceQuote
    from ..models import QuoteErrorRecord

logger = logging.getLogger(__name__)


class QuoteValidator:
    """Handles quote validation with auto-correction and error logging."""

    def __init__(self, auto_accept_threshold: float = 85.0):
        """
        Initialize quote validator.

        Args:
            auto_accept_threshold: Threshold for auto-accepting corrections (%)
        """
        self.corrector = QuoteCorrector(auto_accept_threshold)
        self.console = Console()

    def validate_and_correct_quote(
        self,
        entity_name: str,
        entity_kind: str,
        quote_text: str,
        resource: "Resource",
        quote_error_log: List["QuoteErrorRecord"],
        current_retry: int = 0,
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

        Returns:
            ResourceQuote if successful, None if validation fails
        """
        # First try to create ResourceQuote directly
        try:
            return resource.quote(quote_text)
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
    ) -> Optional["ResourceQuote"]:
        """Handle quote validation failure with correction attempts."""
        from ..models import QuoteErrorRecord

        # Generate correction suggestions
        suggestions = self.corrector.generate_suggestions(quote_text, resource)

        # Check if we can auto-accept the best suggestion
        if suggestions and self.corrector.should_auto_accept(suggestions):
            best_suggestion = self.corrector.get_best_suggestion(suggestions)

            try:
                # Try to use the auto-corrected quote
                corrected_quote = resource.quote(best_suggestion)

                # Log the auto-correction
                self._log_auto_correction(
                    entity_name, quote_text, best_suggestion, suggestions[0][1]
                )

                # Record as resolved error
                error_record = QuoteErrorRecord(
                    entity_name=entity_name,
                    entity_kind=entity_kind,
                    original_quote=quote_text,
                    error_type="not_found",
                    suggested_corrections=[s[0] for s in suggestions[:3]],
                    matched_percentage=suggestions[0][1],
                    final_accepted_quote=best_suggestion,
                    retry_attempt=current_retry,
                    retry_message="Auto-corrected with high confidence",
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
        self._log_quote_failure(entity_name, quote_text, suggestions)

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
        """Classify the type of quote error for logging."""
        from .utilities import detect_split_quote

        # Check if it's a split quote
        if detect_split_quote(quote_text, resource):
            return "split_quote"

        # Check if we have good partial matches
        if suggestions and suggestions[0][1] > 50:
            return "not_found"  # Likely paraphrased or minor differences

        # Check if it's likely a placeholder
        if quote_text.strip() in ["...", ".", ""]:
            return "paraphrased"

        return "not_found"

    def _log_auto_correction(
        self,
        entity_name: str,
        original_quote: str,
        corrected_quote: str,
        confidence: float,
    ) -> None:
        """Log successful auto-correction."""
        self.console.print(
            Panel(
                f"[green]✅ AUTO-CORRECTED[/green] {entity_name}\n"
                f"[yellow]Original:[/yellow] {original_quote[:100]}...\n"
                f"[green]Corrected:[/green] {corrected_quote[:100]}...\n"
                f"[blue]Confidence:[/blue] {confidence:.1f}%",
                title="Quote Auto-Correction",
                border_style="green",
            )
        )

    def _log_quote_failure(
        self, entity_name: str, quote_text: str, suggestions: List[tuple]
    ) -> None:
        """Log quote validation failure with rich formatting."""
        if not suggestions:
            suggestion_text = "[red]No corrections available[/red]"
        else:
            best_suggestion, confidence = suggestions[0]
            suggestion_text = f"[yellow]Best suggestion ({confidence:.1f}%):[/yellow]\n{best_suggestion[:100]}..."

        self.console.print(
            Panel(
                f"[red]❌ QUOTE FAILED[/red] {entity_name}\n"
                f"[yellow]Failed quote:[/yellow] {quote_text[:100]}...\n"
                f"{suggestion_text}",
                title="Quote Validation Error",
                border_style="red",
            )
        )
