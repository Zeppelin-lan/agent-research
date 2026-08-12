"""Extensible, schema-driven tool registration and execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError


class ToolError(RuntimeError):
    """Tool lookup, validation, or execution failure."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    function: Callable[[BaseModel], Any]

    def schema(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "parameters": self.input_model.model_json_schema()}


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> None:
        if tool.name in self._tools:
            raise ToolError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def clone(self) -> "ToolRegistry":
        """Copy registrations so a runtime can inject scoped tools safely."""
        registry = ToolRegistry()
        registry._tools = dict(self._tools)
        return registry

    def has(self, name: str) -> bool:
        return name in self._tools

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        tool = self._tools[name]
        try:
            inputs = tool.input_model.model_validate(arguments)
            return tool.function(inputs)
        except ValidationError as exc:
            raise ToolError(f"Invalid arguments for {name}: {exc}") from exc
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"{name} failed: {exc}") from exc