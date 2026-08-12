"""Provider-independent model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agent_research.core.state import Message, ModelResponse


class ModelError(RuntimeError):
    """A normalized model-provider failure."""


class ModelAdapter(ABC):
    provider: str
    model_name: str

    @abstractmethod
    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        """Generate the next response, optionally containing tool calls."""
