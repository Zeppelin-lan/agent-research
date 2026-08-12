"""Framework-neutral typed state passed between LangGraph nodes."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ToolResult(BaseModel):
    call_id: str
    name: str
    output: Any = None
    error: str | None = None


class ModelUsage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class ModelResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: ModelUsage | None = None
    raw: dict[str, Any] | None = None


class AgentState(TypedDict):
    run_id: str
    user_task: str
    messages: list[Message]
    retrieved_memories: list[str]
    pending_tool_calls: list[ToolCall]
    tool_calls: list[ToolCall]
    tool_results: list[ToolResult]
    current_step: int
    final_answer: str | None
    last_model_response: ModelResponse | None
    errors: list[str]
    total_input_tokens: int
    total_output_tokens: int


def initial_state(run_id: str, task: str) -> AgentState:
    return AgentState(
        run_id=run_id,
        user_task=task,
        messages=[Message(role="user", content=task)],
        retrieved_memories=[],
        pending_tool_calls=[],
        tool_calls=[],
        tool_results=[],
        current_step=0,
        final_answer=None,
        last_model_response=None,
        errors=[],
        total_input_tokens=0,
        total_output_tokens=0,
    )