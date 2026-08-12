"""YAML-configured reproducible batch experiments."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from agent_research.core.agent import AgentRuntime
from agent_research.core.config import ExperimentConfig, RuntimeConfig
from agent_research.memory.local_memory import SQLiteMemory
from agent_research.models.factory import create_model
from agent_research.tools import default_registry

logger = logging.getLogger(__name__)


class Task(BaseModel):
    id: str
    task: str
    expected_contains: str | None = None


def load_tasks(path: Path) -> list[Task]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Tasks file must contain a JSON array")
    return [Task.model_validate(item) for item in data]


def run_experiment(config: ExperimentConfig) -> list[dict[str, Any]]:
    random.seed(config.seed)
    tasks = load_tasks(config.tasks_file)
    memory = SQLiteMemory(config.memory.path) if config.memory.enabled else None
    rows: list[dict[str, Any]] = []
    for model_config in config.models:
        model = create_model(model_config)
        runtime_config = RuntimeConfig(model=model_config, memory=config.memory, max_steps=config.max_steps, trace_dir=config.trace_dir, seed=config.seed)
        for task in tasks:
            started = perf_counter()
            try:
                state = AgentRuntime(model, default_registry(), runtime_config, memory).run(task.task)
                answer = state["final_answer"] or ""
                success = not state["errors"] and (task.expected_contains is None or task.expected_contains.casefold() in answer.casefold())
                error = "; ".join(state["errors"])
            except Exception as exc:
                logger.exception("Experiment run failed")
                state, answer, success, error = None, "", False, str(exc)
            rows.append({"task_id": task.id, "provider": model_config.provider, "model": model_config.model, "success": success, "steps": state["current_step"] if state else 0, "tool_calls": len(state["tool_calls"]) if state else 0, "latency_ms": round((perf_counter() - started) * 1000, 3), "input_tokens": state["total_input_tokens"] if state else 0, "output_tokens": state["total_output_tokens"] if state else 0, "answer": answer, "error": error})
    config.results_file.parent.mkdir(parents=True, exist_ok=True)
    with config.results_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["task_id"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    rows = run_experiment(ExperimentConfig.from_yaml(args.config))
    print(f"Completed {len(rows)} runs")


if __name__ == "__main__":
    main()
