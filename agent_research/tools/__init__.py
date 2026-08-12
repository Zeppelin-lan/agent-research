"""Built-in tools and default registry."""

from agent_research.tools.calculator import CalculatorInput, calculator
from agent_research.tools.datetime_tool import DateTimeInput, current_datetime
from agent_research.tools.document_search import DocumentSearchInput, document_search
from agent_research.tools.kv_lookup import KVLookupInput, kv_lookup
from agent_research.tools.memory_tool import StoreMemoryInput, create_memory_tool
from agent_research.tools.registry import ToolRegistry, ToolSpec


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ToolSpec("calculator", "Evaluate a basic arithmetic expression.", CalculatorInput, calculator))
    registry.register(ToolSpec("current_datetime", "Get the current date and time in a timezone.", DateTimeInput, current_datetime))
    registry.register(ToolSpec("document_search", "Search text files under a local directory.", DocumentSearchInput, document_search))
    registry.register(ToolSpec("kv_lookup", "Look up a key in a local JSON object or JSON file.", KVLookupInput, kv_lookup))
    return registry


__all__ = [
    "StoreMemoryInput",
    "ToolRegistry",
    "ToolSpec",
    "create_memory_tool",
    "default_registry",
]