from agent_research.memory.local_memory import SQLiteMemory


def test_memory_persists_retrieves_and_deletes(tmp_path) -> None:
    path = tmp_path / "memory.db"
    first = SQLiteMemory(path)
    relevant_id = first.add_memory("Paris is the capital of France", {"source": "test"})
    first.add_memory("Whales are mammals")
    second = SQLiteMemory(path)
    results = second.retrieve_memories("France capital", top_k=1)
    assert results[0].id == relevant_id
    assert results[0].metadata == {"source": "test"}
    assert second.delete_memory(relevant_id)
    second.clear()
    assert second.retrieve_memories("anything") == []


def test_memory_does_not_return_zero_score_records(tmp_path) -> None:
    memory = SQLiteMemory(tmp_path / "memory.db")
    memory.add_memory("Whales are mammals")
    assert memory.retrieve_memories("quantum compiler", top_k=5) == []