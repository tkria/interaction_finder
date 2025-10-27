from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, Field


class Term(BaseModel):
    """A search term with optional kind and attributes."""

    kind: Optional[str] = None
    name: str
    attributes: dict[str, str] = Field(default_factory=dict)
