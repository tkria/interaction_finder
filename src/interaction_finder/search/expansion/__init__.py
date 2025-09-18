"""
Query expansion strategies for improving search recall and precision.

This package contains LLM-based query expansion for intelligent, contextual
term generation that adapts to any research domain.
"""

from .llm import LLMQueryExpander, create_llm_expander

__all__ = [
    "LLMQueryExpander",
    "create_llm_expander",
]
