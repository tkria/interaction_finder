"""
Search backend implementations for different academic databases and search engines.

This package contains concrete implementations of the SearchBackend interface
for various academic search services.
"""

from .pubmed import PubMedBackend
from .perplexica import PerplexicaBackend

__all__ = [
    "PubMedBackend",
    "PerplexicaBackend",
]
