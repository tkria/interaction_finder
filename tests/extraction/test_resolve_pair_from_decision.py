"""Tests for _resolve_pair_from_decision error recovery logic."""

import logging
from unittest.mock import MagicMock

import pytest

from interaction_finder.extraction.stages.consolidate_entities import (
    _resolve_pair_from_decision,
)


class TestResolvePairFromDecision:
    """Tests for the LLM decision ID/token resolution fallback logic."""

    def make_pair_info(
        self,
    ) -> tuple[dict[int, tuple[str, str, str]], dict[str, int]]:
        """Create standard test data structures.

        Returns:
            id_to_pair_info: {id: (parent, child, token)}
            token_to_id: {token: id}
        """
        # Standard test pairs
        id_to_pair_info = {
            1: ("ParentA", "ChildA", "tokA"),
            2: ("ParentB", "ChildB", "tokB"),
            3: ("ParentC", "ChildC", "tokC"),
        }
        token_to_id = {"tokA": 1, "tokB": 2, "tokC": 3}
        return id_to_pair_info, token_to_id

    def test_both_id_and_token_valid_perfect_match(self):
        """When ID and token both valid and match, return the pair."""
        id_to_pair_info, token_to_id = self.make_pair_info()
        logger = MagicMock()
        result = _resolve_pair_from_decision(
            decision_id=1,
            decision_token="tokA",
            id_to_pair_info=id_to_pair_info,
            token_to_id=token_to_id,
            logger=logger,
        )
        assert result == ("ParentA", "ChildA")
        # No warnings for perfect match
        logger.warning.assert_not_called()

    def test_id_invalid_token_valid_uses_token(self):
        """When ID is invalid but token is valid, use token and warn."""
        id_to_pair_info, token_to_id = self.make_pair_info()
        logger = MagicMock()
        result = _resolve_pair_from_decision(
            decision_id=999,  # Invalid ID
            decision_token="tokB",  # Valid token
            id_to_pair_info=id_to_pair_info,
            token_to_id=token_to_id,
            logger=logger,
        )
        assert result == ("ParentB", "ChildB")
        logger.warning.assert_called_once()
        warning_msg = logger.warning.call_args[0][0]
        assert "pair_id=999 not found" in warning_msg
        assert "tokB" in warning_msg
        assert "Using token" in warning_msg

    def test_both_invalid_returns_none(self):
        """When both ID and token are invalid, return None and warn."""
        id_to_pair_info, token_to_id = self.make_pair_info()
        logger = MagicMock()
        result = _resolve_pair_from_decision(
            decision_id=999,  # Invalid ID
            decision_token="invalid_tok",  # Invalid token
            id_to_pair_info=id_to_pair_info,
            token_to_id=token_to_id,
            logger=logger,
        )
        assert result is None
        logger.warning.assert_called_once()
        warning_msg = logger.warning.call_args[0][0]
        assert "unresolvable" in warning_msg
        assert "pair_id=999" in warning_msg
        assert "invalid_tok" in warning_msg

    def test_token_mismatch_different_id_prefers_token(self):
        """When token points to different valid pair than ID, prefer token."""
        id_to_pair_info, token_to_id = self.make_pair_info()
        logger = MagicMock()
        # ID 1 has token "tokA", but we provide "tokB" (which maps to ID 2)
        result = _resolve_pair_from_decision(
            decision_id=1,  # Valid ID with token "tokA"
            decision_token="tokB",  # Valid token but for different ID
            id_to_pair_info=id_to_pair_info,
            token_to_id=token_to_id,
            logger=logger,
        )
        # Should use token's pair, not ID's pair
        assert result == ("ParentB", "ChildB")
        logger.warning.assert_called_once()
        warning_msg = logger.warning.call_args[0][0]
        assert "conflict" in warning_msg
        assert "Using token" in warning_msg

    def test_token_invalid_id_valid_trusts_id(self):
        """When token is invalid but ID is valid, trust ID and warn."""
        id_to_pair_info, token_to_id = self.make_pair_info()
        logger = MagicMock()
        result = _resolve_pair_from_decision(
            decision_id=1,  # Valid ID
            decision_token="wrong_tok",  # Invalid token (not in token_to_id)
            id_to_pair_info=id_to_pair_info,
            token_to_id=token_to_id,
            logger=logger,
        )
        # Should use ID's pair
        assert result == ("ParentA", "ChildA")
        logger.warning.assert_called_once()
        warning_msg = logger.warning.call_args[0][0]
        assert "token mismatch" in warning_msg
        assert "expected 'tokA'" in warning_msg
        assert "got 'wrong_tok'" in warning_msg
        assert "Using ID" in warning_msg

    def test_token_maps_to_same_id_with_wrong_token_trusts_id(self):
        """When token is wrong but ID is valid and token doesn't map elsewhere, trust ID."""
        id_to_pair_info, token_to_id = self.make_pair_info()
        logger = MagicMock()
        # Provide wrong token that doesn't map to any valid pair
        result = _resolve_pair_from_decision(
            decision_id=2,  # Valid ID with expected token "tokB"
            decision_token="typo_tok",  # Invalid token
            id_to_pair_info=id_to_pair_info,
            token_to_id=token_to_id,
            logger=logger,
        )
        assert result == ("ParentB", "ChildB")
        logger.warning.assert_called_once()
        warning_msg = logger.warning.call_args[0][0]
        assert "Using ID" in warning_msg

    def test_empty_data_structures(self):
        """Empty lookup dicts should return None for any input."""
        logger = MagicMock()
        result = _resolve_pair_from_decision(
            decision_id=1,
            decision_token="tok",
            id_to_pair_info={},
            token_to_id={},
            logger=logger,
        )
        assert result is None
        logger.warning.assert_called_once()

    def test_logging_includes_all_relevant_info(self):
        """Verify warning messages include diagnostic information."""
        id_to_pair_info, token_to_id = self.make_pair_info()
        logger = MagicMock()
        # Test the conflict case which has the most detailed logging
        _resolve_pair_from_decision(
            decision_id=1,
            decision_token="tokC",  # Maps to ID 3, not 1
            id_to_pair_info=id_to_pair_info,
            token_to_id=token_to_id,
            logger=logger,
        )
        warning_msg = logger.warning.call_args[0][0]
        # Should include: the decision_id, expected token, received token, token's id
        assert "pair_id=1" in warning_msg
        assert "'tokA'" in warning_msg  # Expected token
        assert "'tokC'" in warning_msg  # Received token
        assert "id=3" in warning_msg  # Token's ID
