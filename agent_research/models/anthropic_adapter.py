"""Anthropic Messages API adapter."""

from __future__ import annotations

import os
from typing import Any

from agent_research.core.state import Message, ModelResponse, ModelUsage, ToolCall
from agent_research.models.base import ModelAdapter, ModelError


class AnthropicAdapter(ModelAdapter):
    provider = "anthropic"

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 1024) -> None:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise ModelError("ANTHROPIC_API_KEY is not set")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise ModelError("Install the 'anthropic' project extra") from exc
        self.model_name, self.temperature, self.max_tokens = model_name, temperature, max_tokens
        self.client = Anthropic()

    def generate(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        system = "\n".join(m.content for m in messages if m.role == "system")
        converted: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue
            if message.role == "tool":
                converted.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": message.tool_call_id, "content": message.content}]})
            else:
                converted.append({"role": message.role, "content": message.content})
        try:
            response = self.client.messages.create(
                model=self.model_name, system=system, messages=converted, tools=[{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]} for t in tools],
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            text = "".join(block.text for block in response.content if block.type == "text")
            calls = [ToolCall(id=block.id, name=block.name, arguments=block.input) for block in response.content if block.type == "tool_use"]
            return ModelResponse(content=text, tool_calls=calls, usage=ModelUsage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens))
        except Exception as exc:
            raise ModelError(f"Anthropic request failed: {exc}") from exc
