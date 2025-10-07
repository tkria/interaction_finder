"""
Output formatting module for compliance with Interaction List project requirements.

This module handles the transformation and serialization of extraction results
into the required directory structure and file formats.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from .models import EntityPairOut, BatchExtractionResult


class OutputFormatter:
    """
    Formats extraction results according to Interaction List project specifications.

    Handles:
    - Directory structure: {output_root}/{term}/{mode}/{model}/repeat_{repeat}/
    - Required files: pairs.json, usage.json
    - Pair normalization and deduplication
    - Atomic file writes with idempotency
    """

    def __init__(
        self, output_root: Path, term: str, mode: str, model: str, repeat: int = 1
    ):
        """
        Initialize output formatter.

        Args:
            output_root: Base output directory
            term: Research term (e.g., "BRCA1")
            mode: Research mode (e.g., "basic", "deep")
            model: Model name with provider prefix removed
            repeat: Repeat number starting from 1
        """
        self.output_root = Path(output_root)
        self.term = self._sanitize_path_component(term)
        self.mode = self._sanitize_path_component(mode)
        self.model = self._sanitize_model_name(model)
        self.repeat = repeat

        # Create the output directory path
        self.output_dir = (
            self.output_root
            / self.term
            / self.mode
            / self.model
            / f"repeat_{self.repeat}"
        )

    def _sanitize_path_component(self, component: str) -> str:
        """Sanitize path component for filesystem safety."""
        if not component:
            return "unknown"

        # Replace problematic characters
        sanitized = re.sub(r'[<>:"/\\|?*]', "_", component)
        # Remove leading/trailing dots and spaces
        sanitized = sanitized.strip(". ")
        # Limit length to reasonable filesystem limit
        sanitized = sanitized[:100]

        return sanitized or "unknown"

    def _sanitize_model_name(self, model: str) -> str:
        """Sanitize model name by removing provider prefix."""
        if not model:
            return "unknown"

        # Remove provider prefix (e.g., "openai:" → "")
        if ":" in model:
            model = model.split(":", 1)[1]

        # Apply general sanitization
        return self._sanitize_path_component(model)

    def prepare_directory(self) -> Path:
        """
        Create output directory structure.

        Returns:
            Path to the created output directory
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir

    def is_complete(self) -> bool:
        """
        Check if processing is complete by looking for pairs.json.

        Returns:
            True if pairs.json exists and is valid JSON
        """
        pairs_path = self.output_dir / "pairs.json"
        if not pairs_path.exists():
            return False

        try:
            with open(pairs_path, "r", encoding="utf-8") as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, IOError):
            return False

    def format_pair(self, pair: EntityPairOut) -> Dict[str, Any]:
        """
        Format a single entity pair to the required specification.

        Args:
            pair: EntityPairOut instance to format

        Returns:
            Dictionary formatted per specification
        """
        # Convert confidence score (0-1 float) to categorical
        if pair.confidence >= 0.8:
            confidence = "high"
        elif pair.confidence >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        # Join evidence texts into reasoning
        reasoning = ". ".join(pair.evidence) if pair.evidence else ""

        # Strip angle brackets from all string fields
        first = self._strip_angle_brackets(pair.entity_a.name)
        first_kind = self._strip_angle_brackets(pair.entity_a.kind)
        second = self._strip_angle_brackets(pair.entity_b.name)
        second_kind = self._strip_angle_brackets(pair.entity_b.kind)
        reasoning = self._strip_angle_brackets(reasoning)

        # Create resource list (using source documents as resources)
        resources = [self._strip_angle_brackets(url) for url in pair.source_documents]

        return {
            "first": first,
            "first_kind": first_kind,
            "second": second,
            "second_kind": second_kind,
            "reasoning": reasoning,
            "resources": resources,
            "confidence": confidence,
        }

    def _strip_angle_brackets(self, text: str) -> str:
        """Strip angle brackets from string fields."""
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        return re.sub(r"[<>]", "", text)

    def normalize_pairs(
        self, pairs: List[EntityPairOut], entity_kinds: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Normalize and deduplicate entity pairs.

        Args:
            pairs: List of EntityPairOut instances
            entity_kinds: Ordered list of entity kinds for normalization

        Returns:
            List of normalized, deduplicated pairs
        """
        formatted_pairs = []
        seen_pairs = set()  # For deduplication

        for pair in pairs:
            # Skip pairs with missing required fields
            if not pair.entity_a.name or not pair.entity_b.name:
                continue

            # Format the pair
            formatted_pair = self.format_pair(pair)

            # Normalize entity order based on kind configuration
            normalized_pair = self._normalize_entity_order(formatted_pair, entity_kinds)

            # Create deduplication key (order-insensitive)
            dedup_key = self._create_dedup_key(normalized_pair)

            # Skip duplicates
            if dedup_key in seen_pairs:
                continue

            seen_pairs.add(dedup_key)
            formatted_pairs.append(normalized_pair)

        return formatted_pairs

    def _normalize_entity_order(
        self, pair: Dict[str, Any], entity_kinds: List[str]
    ) -> Dict[str, Any]:
        """
        Normalize entity order based on kind configuration.

        Args:
            pair: Formatted pair dictionary
            entity_kinds: Ordered list of entity kinds

        Returns:
            Pair with normalized entity order
        """
        try:
            first_kind = pair["first_kind"]
            second_kind = pair["second_kind"]

            # If kinds are different, order by entity_kinds precedence
            if first_kind != second_kind:
                first_idx = (
                    entity_kinds.index(first_kind)
                    if first_kind in entity_kinds
                    else 999
                )
                second_idx = (
                    entity_kinds.index(second_kind)
                    if second_kind in entity_kinds
                    else 999
                )

                # If second should come first, swap
                if second_idx < first_idx:
                    return {
                        "first": pair["second"],
                        "first_kind": pair["second_kind"],
                        "second": pair["first"],
                        "second_kind": pair["first_kind"],
                        "reasoning": pair["reasoning"],
                        "resources": pair["resources"],
                        "confidence": pair["confidence"],
                    }

            # For same kinds or when order is already correct, return as-is
            return pair

        except (ValueError, KeyError):
            # If normalization fails, use fallback alphabetical order
            if pair["first"] > pair["second"]:
                return {
                    "first": pair["second"],
                    "first_kind": pair["second_kind"],
                    "second": pair["first"],
                    "second_kind": pair["first_kind"],
                    "reasoning": pair["reasoning"],
                    "resources": pair["resources"],
                    "confidence": pair["confidence"],
                }
            return pair

    def _create_dedup_key(self, pair: Dict[str, Any]) -> str:
        """Create order-insensitive deduplication key."""
        # Sort entities alphabetically for consistent dedup key
        entities = sorted(
            [(pair["first"], pair["first_kind"]), (pair["second"], pair["second_kind"])]
        )
        return f"{entities[0][0]}:{entities[0][1]}|{entities[1][0]}:{entities[1][1]}"

    def save_pairs(self, pairs: List[Dict[str, Any]]) -> None:
        """
        Save entity pairs to pairs.json with atomic write.

        Args:
            pairs: List of formatted pair dictionaries
        """
        pairs_path = self.output_dir / "pairs.json"
        self._save_atomic(pairs_path, pairs)

    def save_usage(self, usage: Dict[str, Any]) -> None:
        """
        Save token usage to usage.json with atomic write.

        Args:
            usage: Token usage dictionary
        """
        usage_path = self.output_dir / "usage.json"
        self._save_atomic(usage_path, usage)

    def save_optional_file(self, filename: str, data: Any) -> None:
        """
        Save optional output file with atomic write.

        Args:
            filename: Name of the file to save
            data: Data to save (will be JSON serialized)
        """
        file_path = self.output_dir / filename
        self._save_atomic(file_path, data)

    def _save_atomic(self, path: Path, data: Any) -> None:
        """
        Save data to file atomically with proper JSON formatting.

        Args:
            path: Target file path
            data: Data to save
        """
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file first
        temp_path = path.with_suffix(f"{path.suffix}.tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(
                    data, f, ensure_ascii=False, indent=2, default=self._json_serializer
                )

            # Atomically move temp file to final location
            temp_path.rename(path)

        except Exception:
            # Clean up temp file on error
            if temp_path.exists():
                temp_path.unlink()
            raise

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for datetime objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_extraction_results(
    result: BatchExtractionResult,
    output_root: Path,
    term: str,
    mode: str,
    model: str,
    repeat: int = 1,
    usage_data: Optional[Dict[str, Any]] = None,
    entity_kinds: Optional[List[str]] = None,
) -> Path:
    """
    Save extraction results using the OutputFormatter.

    Args:
        result: BatchExtractionResult containing extracted pairs
        output_root: Base output directory
        term: Research term
        mode: Research mode
        model: Model name (with provider prefix)
        repeat: Repeat number
        usage_data: Token usage data (optional)
        entity_kinds: List of entity kinds for normalization

    Returns:
        Path to the output directory
    """
    # Create formatter
    formatter = OutputFormatter(output_root, term, mode, model, repeat)

    # Prepare output directory
    output_dir = formatter.prepare_directory()

    # Extract all pairs from group summaries
    # Note: In the current implementation, pairs are stored in state.entity_pairs
    # We'll need to collect them from the result structure
    all_pairs = []

    # For now, we'll need to extract pairs from somewhere in the result
    # This might need to be updated based on how pairs are stored
    # Currently pairs seem to be in the state, not in the result

    # Default entity kinds if not provided
    if entity_kinds is None:
        entity_kinds = ["gene", "disease"]

    # Normalize and save pairs
    normalized_pairs = formatter.normalize_pairs(all_pairs, entity_kinds)
    formatter.save_pairs(normalized_pairs)

    # Save usage data
    if usage_data is None:
        usage_data = {}
    formatter.save_usage(usage_data)

    return output_dir
