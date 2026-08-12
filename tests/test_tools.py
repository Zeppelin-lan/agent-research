import json

import pytest

from agent_research.tools.calculator import CalculatorInput, calculator
from agent_research.tools.document_search import DocumentSearchInput, document_search
from agent_research.tools.kv_lookup import KVLookupInput, kv_lookup
from agent_research.tools.registry import ToolError, ToolRegistry, ToolSpec


def test_registry_schema_and_execution() -> None:
    registry = ToolRegistry()
    registry.register(ToolSpec("calculator", "Arithmetic", CalculatorInput, calculator))
    assert registry.schemas()[0]["parameters"]["properties"]["expression"]
    assert registry.execute("calculator", {"expression": "2 + 3 * 4"}) == 14


def test_registry_rejects_unknown_tool() -> None:
    with pytest.raises(ToolError, match="Unknown tool"):
        ToolRegistry().execute("missing", {})


@pytest.mark.parametrize(("expression", "result"), [("(8-2)/3", 2), ("2**8", 256), ("-4+1", -3)])
def test_calculator(expression: str, result: float) -> None:
    assert calculator(CalculatorInput(expression=expression)) == result


def test_calculator_rejects_code() -> None:
    with pytest.raises(ToolError):
        calculator(CalculatorInput(expression="__import__('os')"))


def test_document_search(tmp_path) -> None:
    (tmp_path / "paper.md").write_text("Agents use tools.\nMemory persists.", encoding="utf-8")
    results = document_search(DocumentSearchInput(query="agent tools", directory=tmp_path))
    assert results[0]["line"] == 1


def test_kv_lookup_inline_and_file(tmp_path) -> None:
    assert kv_lookup(KVLookupInput(key="x", data={"x": 3})) == 3
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"name": "Ada"}), encoding="utf-8")
    assert kv_lookup(KVLookupInput(key="name", file=path)) == "Ada"
