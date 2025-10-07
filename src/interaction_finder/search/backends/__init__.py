"""
Search backend implementations for different academic databases and search engines.

This package contains concrete implementations of the SearchBackend interface
for various academic search services.
"""

from .pubmed import PubMedBackend
from .perplexica import PerplexicaBackend
from .openai_search import OpenAISearchBackend

__all__ = [
    "PubMedBackend",
    "PerplexicaBackend",
    "OpenAISearchBackend",
]
