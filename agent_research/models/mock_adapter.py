"""Deterministic scripted model for tests and examples."""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable

from agent_research.core.state import Message, ModelResponse
from agent_research.models.base import ModelAdapter, ModelError


class MockModelAdapter(ModelAdapter):
    provider = "mock"

    def __init__(
        self, responses: Iterable[ModelResponse] | None = None, model_name: str = "mock-model"
    ) -> None:
        self.model_name = model_name
        self._responses = deque(responses or [ModelResponse(content="Mock final answer")])
        self.calls: list[tuple[list[Message], list[dict[str, Any]]]] = []

    def generate(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        self.calls.append((list(messages), list(tools)))
        if not self._responses:
            raise ModelError("Mock response script exhausted")
        return self._responses.popleft()
