import json

from agent_research.core.agent import AgentRuntime
from agent_research.core.config import MemoryConfig, RuntimeConfig
from agent_research.core.state import ModelResponse, ToolCall
from agent_research.models.mock_adapter import MockModelAdapter
from agent_research.tools import default_registry


def test_calculator_tool_loop_and_trace(tmp_path) -> None:
    model = MockModelAdapter([
        ModelResponse(tool_calls=[ToolCall(id="call-1", name="calculator", arguments={"expression": "6 * 7"})]),
        ModelResponse(content="The answer is 42."),
    ])
    config = RuntimeConfig(memory=MemoryConfig(enabled=False), trace_dir=tmp_path, max_steps=4)
    state = AgentRuntime(model, default_registry(), config).run("What is 6 times 7?", run_id="integration")
    assert state["current_step"] == 2
    assert state["tool_results"][0].output == 42
    assert state["final_answer"] == "The answer is 42."
    assert model.calls[1][0][-1].role == "tool"
    trace = json.loads((tmp_path / "integration.json").read_text(encoding="utf-8"))
    assert [event["event"] for event in trace["events"]] == ["memory_loaded", "model_response", "tool_result", "model_response", "final"]


def test_max_steps_stops_loop(tmp_path) -> None:
    model = MockModelAdapter([ModelResponse(tool_calls=[ToolCall(id="c", name="calculator", arguments={"expression": "1+1"})])])
    config = RuntimeConfig(memory=MemoryConfig(enabled=False), trace_dir=tmp_path, max_steps=1)
    state = AgentRuntime(model, default_registry(), config).run("loop")
    assert "max_steps=1" in (state["final_answer"] or "")
    assert not state["tool_results"]
