"""OpenAI Responses API adapter with function calling."""

from __future__ import annotations

import json
import os
from typing import Any

from agent_research.core.state import Message, ModelResponse, ModelUsage, ToolCall
from agent_research.models.base import ModelAdapter, ModelError


class OpenAIAdapter(ModelAdapter):
    provider = "openai"

    def __init__(
        self,
        model_name: str,
        temperature: float | None = None,
        max_output_tokens: int = 1024,
        client: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        if client is not None:
            self.client = client
            return
        if not os.getenv("OPENAI_API_KEY"):
            raise ModelError("OPENAI_API_KEY is not set")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ModelError("Install the 'openai' project extra") from exc
        self.client = OpenAI()

    @staticmethod
    def _response_input(messages: list[Message]) -> list[dict[str, Any]]:
        """Translate neutral history into Responses API input items."""
        items: list[dict[str, Any]] = []
        for message in messages:
            if message.provider_items:
                items.extend(message.provider_items)
                continue
            if message.role == "tool":
                if not message.tool_call_id:
                    raise ModelError("OpenAI tool result is missing tool_call_id")
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": message.tool_call_id,
                        "output": message.content,
                    }
                )
                continue
            if message.content:
                items.append({"role": message.role, "content": message.content})
            for call in message.tool_calls:
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.id,
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    }
                )
        return items

    @staticmethod
    def _response_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
            for tool in tools
        ]

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> ModelResponse:
        request: dict[str, Any] = {
            "model": self.model_name,
            "input": self._response_input(messages),
            "max_output_tokens": self.max_output_tokens,
        }
        response_tools = self._response_tools(tools)
        if response_tools:
            request["tools"] = response_tools
            request["tool_choice"] = "auto"
        if self.temperature is not None:
            request["temperature"] = self.temperature

        try:
            response = self.client.responses.create(**request)
            output = [item.model_dump(mode="json", exclude_none=True) for item in response.output]
            calls = [
                ToolCall(
                    id=item.call_id,
                    name=item.name,
                    arguments=json.loads(item.arguments or "{}"),
                )
                for item in response.output
                if item.type == "function_call"
            ]
            usage = response.usage
            return ModelResponse(
                content=response.output_text or "",
                tool_calls=calls,
                usage=ModelUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                )
                if usage
                else None,
                raw={"response_id": response.id, "output": output},
            )
        except ModelError:
            raise
        except Exception as exc:
            raise ModelError(f"OpenAI Responses request failed: {exc}") from exc
