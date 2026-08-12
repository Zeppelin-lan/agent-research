"""Google Gemini adapter."""

from __future__ import annotations

import os
import uuid
from typing import Any

from agent_research.core.state import Message, ModelResponse, ModelUsage, ToolCall
from agent_research.models.base import ModelAdapter, ModelError


class GeminiAdapter(ModelAdapter):
    provider = "gemini"

    def __init__(self, model_name: str, temperature: float = 0.0, max_tokens: int = 1024) -> None:
        if not os.getenv("GEMINI_API_KEY"):
            raise ModelError("GEMINI_API_KEY is not set")
        try:
            from google import genai
        except ImportError as exc:
            raise ModelError("Install the 'gemini' project extra") from exc
        self.model_name, self.temperature, self.max_tokens = model_name, temperature, max_tokens
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate(self, messages: list[Message], tools: list[dict[str, Any]]) -> ModelResponse:
        try:
            from google.genai import types
            contents = "\n".join(f"{m.role}: {m.content}" for m in messages)
            declarations = [types.FunctionDeclaration(name=t["name"], description=t["description"], parameters_json_schema=t["parameters"]) for t in tools]
            response = self.client.models.generate_content(
                model=self.model_name, contents=contents,
                config=types.GenerateContentConfig(temperature=self.temperature, max_output_tokens=self.max_tokens, tools=[types.Tool(function_declarations=declarations)]),
            )
            calls = [ToolCall(id=str(uuid.uuid4()), name=part.function_call.name, arguments=dict(part.function_call.args)) for part in response.candidates[0].content.parts if part.function_call]
            usage = response.usage_metadata
            return ModelResponse(content=response.text or "", tool_calls=calls, usage=ModelUsage(input_tokens=getattr(usage, "prompt_token_count", None), output_tokens=getattr(usage, "candidates_token_count", None)))
        except Exception as exc:
            raise ModelError(f"Gemini request failed: {exc}") from exc
