"""Run one agent task from the command line."""

import argparse
import logging

from agent_research.core.agent import AgentRuntime
from agent_research.core.config import ModelConfig, RuntimeConfig
from agent_research.memory.local_memory import SQLiteMemory
from agent_research.models.factory import create_model
from agent_research.tools import default_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini", "mock"], default="mock")
    parser.add_argument("--model", default="mock-model")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    model_config = ModelConfig(provider=args.provider, model=args.model)
    config = RuntimeConfig(model=model_config, max_steps=args.max_steps)
    memory = SQLiteMemory(config.memory.path) if config.memory.enabled else None
    state = AgentRuntime(create_model(model_config), default_registry(), config, memory).run(args.task)
    print(state["final_answer"])


if __name__ == "__main__":
    main()
