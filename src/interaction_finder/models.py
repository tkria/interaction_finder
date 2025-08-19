from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Any, Optional

from pydantic import BaseModel, Field

class Term(BaseModel):
    kind: Optional[str] = None
    name: str
    attributes: dict[str, str] = Field(default_factory=dict)
