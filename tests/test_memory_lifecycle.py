import json

from agent_research.core.agent import AgentRuntime
from agent_research.core.config import MemoryConfig, RuntimeConfig
from agent_research.core.state import ModelResponse, ToolCall
from agent_research.memory.local_memory import SQLiteMemory
from agent_research.models.mock_adapter import MockModelAdapter
from agent_research.tools import create_memory_tool, default_registry


def test_store_memory_tool_writes_existing_sqlite_backend(tmp_path) -> None:
    path = tmp_path / "memory.db"
    memory = SQLiteMemory(path)
    registry = default_registry()
    registry.register(create_memory_tool(memory, lambda: "tool-run"))

    output = registry.execute(
        "store_memory",
        {
            "text": "User prefers concise answers.",
            "kind": "preference",
            "key": "response_style",
        },
    )

    reopened = SQLiteMemory(path)
    records = reopened.retrieve_memories("concise answers")
    assert records[0].id == output["memory_id"]
    assert records[0].metadata == {
        "key": "response_style",
        "kind": "preference",
        "run_id": "tool-run",
        "source": "agent_tool",
    }


def test_two_phase_write_reset_and_retrieve(tmp_path) -> None:
    memory_path = tmp_path / "memory.db"
    trace_dir = tmp_path / "traces"
    config = RuntimeConfig(
        memory=MemoryConfig(enabled=True, top_k=5, path=memory_path),
        trace_dir=trace_dir,
        max_steps=4,
    )

    write_model = MockModelAdapter(
        [
            ModelResponse(
                tool_calls=[
                    ToolCall(
                        id="remember-1",
                        name="store_memory",
                        arguments={
                            "text": "I prefer concise answers.",
                            "kind": "preference",
                            "key": "response_style",
                        },
                    )
                ]
            ),
            ModelResponse(content="I will remember that preference."),
        ]
    )
    runtime_a = AgentRuntime(
        write_model,
        default_registry(),
        config,
        SQLiteMemory(memory_path),
    )
    state_a = runtime_a.run(
        "Remember that I prefer concise answers.",
        run_id="phase-a",
    )

    assert state_a["final_answer"] == "I will remember that preference."
    assert state_a["tool_calls"][0].name == "store_memory"
    write_trace = json.loads((trace_dir / "phase-a.json").read_text(encoding="utf-8"))
    memory_event = next(event for event in write_trace["events"] if event["event"] == "memory_written")
    assert memory_event["stored_text"] == "I prefer concise answers."
    assert memory_event["memory_metadata"]["run_id"] == "phase-a"
    assert memory_event["memory_id"]

    read_model = MockModelAdapter(
        [ModelResponse(content="I should format your answers concisely.")]
    )
    runtime_b = AgentRuntime(
        read_model,
        default_registry(),
        config,
        SQLiteMemory(memory_path),
    )
    state_b = runtime_b.run(
        "How should you format my answers?",
        run_id="phase-b",
    )

    assert state_b["retrieved_memories"] == ["I prefer concise answers."]
    assert state_b["current_step"] == 1
    assert state_b["tool_calls"] == []
    assert state_b["tool_results"] == []
    assert all(
        message.content != "Remember that I prefer concise answers."
        for message in state_b["messages"]
    )
    phase_b_model_messages = read_model.calls[0][0]
    assert "I prefer concise answers." in phase_b_model_messages[0].content
    assert [message.role for message in phase_b_model_messages] == ["system", "user"]
    assert phase_b_model_messages[1].content == "How should you format my answers?"

    read_trace = json.loads((trace_dir / "phase-b.json").read_text(encoding="utf-8"))
    load_event = next(event for event in read_trace["events"] if event["event"] == "memory_loaded")
    assert load_event["retrieved_memories"] == ["I prefer concise answers."]


def test_memory_tool_is_not_registered_when_memory_is_disabled(tmp_path) -> None:
    config = RuntimeConfig(
        memory=MemoryConfig(enabled=False),
        trace_dir=tmp_path,
    )
    model = MockModelAdapter([ModelResponse(content="done")])
    runtime = AgentRuntime(model, default_registry(), config, SQLiteMemory(tmp_path / "unused.db"))
    assert "store_memory" not in {schema["name"] for schema in runtime.tools.schemas()}