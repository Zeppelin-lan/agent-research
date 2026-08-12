"""OpenAI Responses-style chat adapter with tool calling."""

from __future__ import annotations

import json
import os
from typing import Any

from agent_research.core.state import Message, ModelResponse, ModelUsage, ToolCall
from agent_research.models.base import ModelAdapter, ModelError


class OpenAIAdapter(ModelAdapter):
    provider = "openai"

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 1024) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise ModelError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelError("Install the 'openai' project extra") from exc
        self.model_name, self.temperature, self.max_tokens = model_name, temperature, max_tokens
        self.client = OpenAI()

    def generate(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        try:
            provider_messages = []
            for message in messages:
                item: dict[str, Any] = {"role": message.role, "content": message.content}
                if message.name is not None:
                    item["name"] = message.name
                if message.tool_call_id is not None:
                    item["tool_call_id"] = message.tool_call_id
                if message.tool_calls:
                    item["tool_calls"] = [{"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}} for c in message.tool_calls]
                provider_messages.append(item)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=provider_messages,
                tools=[{"type": "function", "function": t} for t in tools],
                tool_choice="auto",
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            choice = response.choices[0].message
            calls = [
                ToolCall(id=c.id, name=c.function.name, arguments=json.loads(c.function.arguments or "{}"))
                for c in (choice.tool_calls or [])
            ]
            usage = response.usage
            return ModelResponse(
                content=choice.content or "", tool_calls=calls,
                usage=ModelUsage(input_tokens=usage.prompt_tokens, output_tokens=usage.completion_tokens)
                if usage else None,
            )
        except Exception as exc:
            raise ModelError(f"OpenAI request failed: {exc}") from exc