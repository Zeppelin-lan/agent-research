"""Readable LangGraph agent loop with explicit memory/model/tool nodes."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Literal

from langgraph.graph import END, START, StateGraph

from agent_research.core.config import RuntimeConfig
from agent_research.core.state import AgentState, Message, initial_state
from agent_research.core.tracing import JsonTracer, Timer
from agent_research.memory.base import MemoryStore
from agent_research.models.base import ModelAdapter, ModelError
from agent_research.tools.registry import ToolError, ToolRegistry
from agent_research.tools.memory_tool import create_memory_tool

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Coordinates replaceable components while exposing the loop as graph nodes."""

    def __init__(self, model: ModelAdapter, tools: ToolRegistry, config: RuntimeConfig, memory: MemoryStore | None = None) -> None:
        self.model, self.tools, self.config, self.memory = model, tools.clone(), config, memory
        self._active_run_id: str | None = None
        if config.memory.enabled and memory is not None:
            self.tools.register(create_memory_tool(memory, self._current_run_id))
        self._tracer: JsonTracer | None = None
        self.graph = self._build_graph()

    def _current_run_id(self) -> str:
        if self._active_run_id is None:
            raise RuntimeError("store_memory can only execute during an agent run")
        return self._active_run_id

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("load_memory", self._load_memory)
        graph.add_node("call_model", self._call_model)
        graph.add_node("execute_tools", self._execute_tools)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "load_memory")
        graph.add_edge("load_memory", "call_model")
        graph.add_conditional_edges("call_model", self._route_model, {"tools": "execute_tools", "finalize": "finalize"})
        graph.add_conditional_edges("execute_tools", self._route_tools, {"model": "call_model", "finalize": "finalize"})
        graph.add_edge("finalize", END)
        return graph.compile()

    def _load_memory(self, state: AgentState) -> dict:
        memories = []
        if self.config.memory.enabled and self.memory is not None:
            memories = [record.text for record in self.memory.retrieve_memories(state["user_task"], self.config.memory.top_k)]
        messages = list(state["messages"])
        context = "Relevant persistent memories from previous independent runs:\n" + "\n".join(f"- {item}" for item in memories) if memories else "No relevant persistent memories were retrieved."
        memory_instruction = (
            " Use store_memory only when the user clearly intends information to persist, "
            "such as an explicit remember request, stable preference, profile fact, or lasting "
            "instruction. Do not store ordinary task content."
            if self.tools.has("store_memory")
            else ""
        )
        messages.insert(0, Message(role="system", content="You are a tool-using research agent. Use tools when useful." + memory_instruction + " " + context))
        assert self._tracer
        self._tracer.record("memory_loaded", 0, retrieved_memories=memories)
        return {"retrieved_memories": memories, "messages": messages}

    def _call_model(self, state: AgentState) -> dict:
        step = state["current_step"] + 1
        try:
            with Timer() as timer:
                response = self.model.generate(state["messages"], self.tools.schemas())
            usage = response.usage
            messages = list(state["messages"])
            provider_items = response.raw.get("output", []) if response.raw else []
            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls,
                    provider_items=provider_items,
                )
            )
            assert self._tracer
            self._tracer.record("model_response", step, timer.elapsed_ms, input_messages=[m.model_dump(mode="json") for m in state["messages"]], retrieved_memories=state["retrieved_memories"], available_tools=self.tools.schemas(), model_response=response.model_dump(mode="json"))
            answer = response.content if not response.tool_calls else None
            return {"current_step": step, "last_model_response": response, "pending_tool_calls": response.tool_calls, "messages": messages, "final_answer": answer, "total_input_tokens": state["total_input_tokens"] + (usage.input_tokens or 0 if usage else 0), "total_output_tokens": state["total_output_tokens"] + (usage.output_tokens or 0 if usage else 0)}
        except ModelError as exc:
            logger.exception("Model call failed")
            assert self._tracer
            self._tracer.record("model_error", step, error=str(exc))
            return {"current_step": step, "errors": [*state["errors"], str(exc)], "final_answer": f"Agent stopped after a model error: {exc}", "pending_tool_calls": []}

    def _route_model(self, state: AgentState) -> Literal["tools", "finalize"]:
        if state["errors"] or not state["pending_tool_calls"] or state["current_step"] >= self.config.max_steps:
            return "finalize"
        return "tools"

    def _execute_tools(self, state: AgentState) -> dict:
        results, messages = list(state["tool_results"]), list(state["messages"])
        errors = list(state["errors"])
        for call in state["pending_tool_calls"]:
            error = None
            try:
                with Timer() as timer:
                    output = self.tools.execute(call.name, call.arguments)
            except ToolError as exc:
                output, error = None, str(exc)
            result_text = json.dumps(output if error is None else {"error": error}, default=str)
            from agent_research.core.state import ToolResult
            results.append(ToolResult(call_id=call.id, name=call.name, output=output, error=error))
            messages.append(Message(role="tool", name=call.name, tool_call_id=call.id, content=result_text))
            assert self._tracer
            self._tracer.record("tool_result", state["current_step"], timer.elapsed_ms if error is None else 0.0, selected_tool=call.name, tool_arguments=call.arguments, tool_output=output, error=error)
            if call.name == "store_memory" and error is None and isinstance(output, dict):
                self._tracer.record(
                    "memory_written",
                    state["current_step"],
                    memory_id=str(output["memory_id"]),
                    stored_text=str(output["text"]),
                    memory_metadata=output["metadata"],
                )
        return {"tool_calls": [*state["t…1154 tokens truncated…["system", "user", "assistant", "tool"]
    content: str = ""
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    provider_items: list[dict[str, Any]] = Field(default_factory=list)


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
