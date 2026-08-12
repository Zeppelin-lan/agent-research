import json
from types import SimpleNamespace
from typing import Any

from agent_research.core.agent import AgentRuntime
from agent_research.core.config import MemoryConfig, RuntimeConfig
from agent_research.core.state import Message
from agent_research.models.openai_adapter import OpenAIAdapter
from agent_research.tools import default_registry


class FakeResponseItem:
    def __init__(self, **data: Any) -> None:
        self._data = data
        for key, value in data.items():
            setattr(self, key, value)

    def model_dump(self, **_: Any) -> dict[str, Any]:
        return dict(self._data)


class FakeResponses:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = iter(responses)
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> Any:
        self.requests.append(request)
        return next(self._responses)


def fake_response(
    response_id: str,
    output: list[FakeResponseItem],
    output_text: str,
    input_tokens: int,
    output_tokens: int,
) -> Any:
    return SimpleNamespace(
        id=response_id,
        output=output,
        output_text=output_text,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def test_responses_api_request_and_function_call_parsing() -> None:
    function_call = FakeResponseItem(
        type="function_call",
        id="fc_1",
        call_id="call_1",
        name="calculator",
        arguments=json.dumps({"expression": "2 + 2"}),
        status="completed",
    )
    responses = FakeResponses(
        [fake_response("resp_1", [function_call], "", 11, 4)]
    )
    client = SimpleNamespace(responses=responses)
    adapter = OpenAIAdapter("gpt-5.6", client=client)

    result = adapter.generate(
        [
            Message(role="system", content="Use tools when useful."),
            Message(role="user", content="What is 2 + 2?"),
        ],
        [
            {
                "name": "calculator",
                "description": "Evaluate arithmetic.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            }
        ],
    )

    request = responses.requests[0]
    assert request["model"] == "gpt-5.6"
    assert request["max_output_tokens"] == 1024
    assert "max_tokens" not in request
    assert "max_completion_tokens" not in request
    assert "messages" not in request
    assert "temperature" not in request
    assert request["tools"][0]["name"] == "calculator"
    assert "function" not in request["tools"][0]
    assert result.tool_calls[0].id == "call_1"
    assert result.tool_calls[0].arguments == {"expression": "2 + 2"}
    assert result.usage is not None
    assert result.usage.input_tokens == 11
    assert result.usage.output_tokens == 4
    assert result.raw == {
        "response_id": "resp_1",
        "output": [function_call.model_dump()],
    }


def test_responses_api_preserves_call_item_before_tool_output() -> None:
    function_call_data = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "calculator",
        "arguments": json.dumps({"expression": "2 + 2"}),
        "status": "completed",
    }
    responses = FakeResponses(
        [fake_response("resp_2", [], "The answer is 4.", 15, 6)]
    )
    client = SimpleNamespace(responses=responses)
    adapter = OpenAIAdapter("gpt-5.6", client=client)

    result = adapter.generate(
        [
            Message(role="user", content="What is 2 + 2?"),
            Message(
                role="assistant",
                provider_items=[function_call_data],
            ),
            Message(
                role="tool",
                name="calculator",
                tool_call_id="call_1",
                content="4",
            ),
        ],
        [],
    )

    request_input = responses.requests[0]["input"]
    assert request_input == [
        {"role": "user", "content": "What is 2 + 2?"},
        function_call_data,
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "4",
        },
    ]
    assert result.content == "The answer is 4."


def test_runtime_completes_responses_function_call_round_trip(tmp_path) -> None:
    function_call = FakeResponseItem(
        type="function_call",
        id="fc_1",
        call_id="call_1",
        name="calculator",
        arguments=json.dumps({"expression": "6 * 7"}),
        status="completed",
    )
    final_message = FakeResponseItem(
        type="message",
        id="msg_1",
        role="assistant",
        status="completed",
        content=[],
    )
    responses = FakeResponses(
        [
            fake_response("resp_1", [function_call], "", 11, 4),
            fake_response(
                "resp_2",
                [final_message],
                "The answer is 42.",
                18,
                6,
            ),
        ]
    )
    adapter = OpenAIAdapter(
        "gpt-5.6",
        client=SimpleNamespace(responses=responses),
    )
    config = RuntimeConfig(
        memory=MemoryConfig(enabled=False),
        trace_dir=tmp_path,
        max_steps=4,
    )

    state = AgentRuntime(
        adapter,
        default_registry(),
        config,
    ).run("What is 6 times 7?", run_id="openai-round-trip")

    assert state["final_answer"] == "The answer is 42."
    assert state["tool_results"][0].output == 42
    second_input = responses.requests[1]["input"]
    assert function_call.model_dump() in second_input
    assert {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": "42",
    } in second_input
