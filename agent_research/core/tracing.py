"""Framework-independent JSON execution traces."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    run_id: str
    timestamp: str
    step: int
    event: str
    provider: str
    model: str
    input_messages: list[dict[str, Any]] = Field(default_factory=list)
    retrieved_memories: list[str] = Field(default_factory=list)
    available_tools: list[dict[str, Any]] = Field(default_factory=list)
    model_response: dict[str, Any] | None = None
    selected_tool: str | None = None
    tool_arguments: dict[str, Any] | None = None
    tool_output: Any = None
    memory_id: str | None = None
    stored_text: str | None = None
    memory_metadata: dict[str, Any] | None = None
    latency_ms: float = 0.0
    final_answer: str | None = None
    error: str | None = None


class RunTrace(BaseModel):
    schema_version: str = "1.0"
    run_id: str
    started_at: str
    finished_at: str | None = None
    task: str
    provider: str
    model: str
    events: list[TraceEvent] = Field(default_factory=list)
    final_answer: str | None = None
    errors: list[str] = Field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class JsonTracer:
    def __init__(self, trace_dir: str | Path, run_id: str, task: str, provider: str, model: str) -> None:
        self.trace_dir = Path(trace_dir)
        self.trace = RunTrace(run_id=run_id, started_at=self.now(), task=task, provider=provider, model=model)
        self._lock = threading.Lock()

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def record(self, event: str, step: int, latency_ms: float = 0.0, **details: Any) -> None:
        with self._lock:
            self.trace.events.append(TraceEvent(run_id=self.trace.run_id, timestamp=self.now(), step=step, event=event, provider=self.trace.provider, model=self.trace.model, latency_ms=round(latency_ms, 3), **details))

    def finish(self, final_answer: str | None, errors: list[str], input_tokens: int, output_tokens: int) -> Path:
        with self._lock:
            self.trace.finished_at = self.now()
            self.trace.final_answer = final_answer
            self.trace.errors = errors
            self.trace.input_tokens = input_tokens
            self.trace.output_tokens = output_tokens
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            path = self.trace_dir / f"{self.trace.run_id}.json"
            path.write_text(self.trace.model_dump_json(indent=2), encoding="utf-8")
            return path


class Timer:
    def __enter__(self) -> "Timer":
        self.started = perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_ms = (perf_counter() - self.started) * 1000