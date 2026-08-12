"""Runtime-injected tool for explicit long-term memory writes."""

from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel, Field

from agent_research.memory.base import MemoryStore
from agent_research.tools.registry import ToolSpec


class StoreMemoryInput(BaseModel):
    text: str = Field(min_length=1, max_length=10_000)
    kind: Literal["preference", "profile", "instruction", "fact"] = "preference"
    key: str | None = Field(default=None, max_length=200)


def create_memory_tool(
    memory_store: MemoryStore,
    run_id_provider: Callable[[], str],
) -> ToolSpec:
    """Bind a store_memory tool to one replaceable memory backend and runtime."""

    def store_memory(inputs: StoreMemoryInput) -> dict[str, object]:
        metadata: dict[str, object] = {
            "kind": inputs.kind,
            "source": "agent_tool",
            "run_id": run_id_provider(),
        }
        if inputs.key is not None:
            metadata["key"] = inputs.key
        memory_id = memory_store.add_memory(inputs.text, metadata)
        return {
            "memory_id": memory_id,
            "text": inputs.text,
            "metadata": metadata,
        }

    return ToolSpec(
        name="store_memory",
        description=(
            "Persist information for future independent runs. Use only for explicit "
            "remember requests, stable preferences, profile facts, or lasting instructions; "
            "do not store ordinary task content."
        ),
        input_model=StoreMemoryInput,
        function=store_memory,
    )