"""Long-term memory interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class MemoryRecord(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None
    created_at: str


class MemoryStore(ABC):
    @abstractmethod
    def add_memory(self, text: str, metadata: dict[str, Any] | None = None) -> str: ...

    @abstractmethod
    def retrieve_memories(self, query: str, top_k: int = 5) -> list[MemoryRecord]: ...

    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool: ...

    @abstractmethod
    def clear(self) -> None: ...
