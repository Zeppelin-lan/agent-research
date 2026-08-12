"""Persistent SQLite memory with lightweight lexical vector retrieval."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent_research.memory.base import MemoryRecord, MemoryStore


def _vector(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9]+", text.casefold()))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    denominator = math.sqrt(sum(v * v for v in left.values()) * sum(v * v for v in right.values()))
    return sum(value * right[token] for token, value in left.items()) / denominator if denominator else 0.0


class SQLiteMemory(MemoryStore):
    """A zero-service baseline; replace with an embedding backend via MemoryStore."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, text TEXT NOT NULL, metadata TEXT NOT NULL, created_at TEXT NOT NULL)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def add_memory(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        memory_id = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute("INSERT INTO memories VALUES (?, ?, ?, ?)", (memory_id, text, json.dumps(metadata or {}, sort_keys=True), created_at))
        return memory_id

    def retrieve_memories(self, query: str, top_k: int = 5) -> list[MemoryRecord]:
        with self._connect() as connection:
            rows = connection.execute("SELECT id, text, metadata, created_at FROM memories").fetchall()
        query_vector = _vector(query)
        records = [MemoryRecord(id=row[0], text=row[1], metadata=json.loads(row[2]), created_at=row[3], score=_cosine(query_vector, _vector(row[1]))) for row in rows]
        relevant = [record for record in records if (record.score or 0) > 0]
        return sorted(relevant, key=lambda record: (-(record.score or 0), record.created_at, record.id))[:top_k]

    def delete_memory(self, memory_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM memories")