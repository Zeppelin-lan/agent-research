# Agent Research

This project is based on today's widely adopted agent architectures and is designed as a research platform for studying LLM agents.

Agent Research is a clean, inspectable Python platform for experiments with a single tool-using LLM agent. Built around LangGraph, it prioritizes modularity, observability, reproducibility, and replaceable components. It intentionally has no UI, managed vector database, multi-agent layer, or unrelated product infrastructure.

## Concepts

- **LLM:** a model that predicts a response and may request structured tool calls.
- **Agent:** the model plus the capabilities and policy context made available to it.
- **Tool:** a named callable with a description and Pydantic-generated JSON input schema.
- **Memory:** information persisted across independent runs and retrieved for a task.
- **Agent state:** typed, short-lived data for one run: messages, calls, results, steps, and answer.
- **Agent runtime:** the LangGraph control flow that moves state through model and tool nodes.
- **Experiment harness:** repeatable batch execution across tasks and model configurations.

## Architecture

```text
                         Experiment harness
                   (YAML -> tasks -> CSV metrics)
                                  |
                                  v
+-----------+    +-------------------------------------------+    +-------------+
| User task | -> | Agent runtime (explicit LangGraph nodes)  | -> | JSON trace  |
+-----------+    +-------------------------------------------+    +-------------+
                    |          |          |          |
                    v          v          v          v
                Model       Tools      State      Memory
                adapter    registry   (typed)    interface
                   |                                  |
          +--------+--------+                         v
          |        |        |                    SQLite +
        OpenAI  Anthropic Gemini                lexical vectors
          \        |       /                         (local)
           \---- Mock ----/                          
             (tests)
```

The main modules are deliberately independent:

- `agent_research/models/`: provider-neutral `ModelAdapter`, three provider adapters, and deterministic mock.
- `agent_research/tools/`: schema-driven registry, four ordinary built-ins, and a runtime-injected memory tool.
- `agent_research/core/agent.py`: explicit runtime nodes, routes, and step limit.
- `agent_research/core/state.py`: framework-neutral Pydantic values and typed graph state.
- `agent_research/memory/`: replaceable interface and persistent SQLite baseline.
- `agent_research/core/tracing.py`: versioned JSON trace models and writer.
- `agent_research/experiments/`: YAML configuration, batch runner, and CSV metrics.

## Agent execution flow

The graph is intentionally visible in `core/agent.py`:

```text
START -> load_memory -> call_model
                            |
                 +----------+----------+
                 | tool calls?         | final/error/limit
                 v                     v
            execute_tools --------> finalize -> END
                 |
                 +-------------> call_model
```

`load_memory` retrieves relevant records once and builds system context. `call_model` passes the full message history and every registered tool schema to the adapter. The model—not application code—selects tools. `execute_tools` validates arguments, executes each requested tool, and appends results as tool messages. The cycle continues until a final model response, an error, or `max_steps`.

## Setup

Python 3.11 or newer is required.

```powershell
cd agent_research
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Install a provider extra as needed:

```powershell
python -m pip install -e ".[openai,dev]"
python -m pip install -e ".[anthropic,dev]"
python -m pip install -e ".[gemini,dev]"
# Or all providers:
python -m pip install -e ".[all]"
```

Copy `.env.example` to `.env` or export the relevant variable. The project never reads keys from source files:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Gemini: `GEMINI_API_KEY`

The CLI defaults to the free deterministic mock. A `.env` loader is intentionally not implicit; use your shell, experiment environment, or a secrets manager to export variables.

## Run one task

```powershell
agent-run "What is 17 * 23?" --provider openai --model gpt-5.6
```

The OpenAI adapter uses the current Responses API (`client.responses.create`),
including flat function-tool definitions, `function_call_output` items, and the
`max_output_tokens` request field. `temperature` is omitted unless configured
explicitly, which keeps the default compatible with reasoning models.

For a no-key smoke test:

```powershell
agent-run "Explain agent state" --provider mock
```

Each run writes `traces/<run-id>.json`. A trace contains timestamps, step numbers, provider/model, model inputs and response, retrieved memory, available tools, requested arguments, tool outputs, latency, token usage when the provider supplies it, errors, and the final answer. The schema is plain Pydantic/JSON and does not depend on LangGraph or a provider tracing service.

## Tools

Built-ins are:

- `calculator`: safely evaluates arithmetic through an AST allowlist (never `eval`).
- `current_datetime`: returns an ISO timestamp for an IANA timezone.
- `document_search`: deterministic line search across local `.txt` and `.md` files.
- `kv_lookup`: reads a key from inline data or a local JSON object.
- `store_memory`: explicitly persists stable information for future independent runs. It is registered only when an enabled `MemoryStore` is injected.

To add a tool:

1. Define a Pydantic input model.
2. Write a callable accepting that model.
3. Register a `ToolSpec(name, description, input_model, function)`.

The registry derives JSON Schema from Pydantic and validates every call before execution. Tools are not selected by conditionals in the runtime.

## Memory

`MemoryStore` defines `add_memory`, `retrieve_memories`, `delete_memory`, and `clear`. `SQLiteMemory` persists text and metadata in a local database, then ranks records using a deterministic bag-of-words cosine vector. This is a reproducible, zero-service baseline—not a semantic embedding model. To use learned embeddings or another local vector store, implement `MemoryStore` and inject it into `AgentRuntime`; no graph change is needed.

Zero-similarity records are excluded from retrieval, even when `top_k` has remaining capacity. This keeps unrelated persistent records out of model context.

## Persistent Memory Lifecycle

Long-term writes are explicit model actions, not automatic conversation logging:

```text
User interaction
      |
      v
LLM selects store_memory
      |
      v
MemoryStore.add_memory(text, metadata)
      |
      v
SQLite persists the record
      |
      v
Run ends; AgentState is discarded
      |
      v
New independent run -> _load_memory() -> retrieve_memories()
      |
      v
Relevant memory enters the new model context
```

`AgentState` is short-term and newly constructed by `initial_state()` for every `AgentRuntime.run()`. Messages, pending calls, completed tool calls/results, step count, and final answer never cross run boundaries. `MemoryStore` is long-term: its SQLite records survive new state objects, new `AgentRuntime` objects, and new processes that use the same database path.

When memory is enabled, `AgentRuntime` clones the supplied `ToolRegistry` and injects a `store_memory` tool bound to that runtime's `MemoryStore`. The model is instructed to use it only for explicit remember requests, stable preferences, profile facts, or lasting instructions. A successful write stores `kind`, optional `key`, `source: agent_tool`, and the active `run_id` as metadata. Ordinary user messages and task content are never persisted automatically.

Every write produces both the normal `tool_result` event and a dedicated `memory_written` event in `traces/<run-id>.json`. The latter contains the memory ID, stored text, metadata, run ID, and graph step. The next run's `memory_loaded` event lists the memories retrieved for that task.

To inspect traces in PowerShell:

```powershell
Get-ChildItem traces\*.json
Get-Content traces\<run-id>.json
```

To inspect the SQLite records without changing them:

```powershell
python -c "import sqlite3; print(sqlite3.connect('memory.sqlite3').execute('select id, text, metadata, created_at from memories').fetchall())"
```

### Manual OpenAI two-process smoke test

Install the OpenAI extra, export `OPENAI_API_KEY`, and run both commands from the project root so they share `memory.sqlite3`:

```powershell
agent-run "Please remember that I prefer concise answers in future conversations." --provider openai --model <configured-model>
agent-run "What response style do I prefer?" --provider openai --model <configured-model>
```

These are separate processes. The second process can receive the preference only through `_load_memory()` reading the persistent SQLite backend; it has no access to the first process's message history. This paid-provider path is intentionally excluded from the default test suite.

## Add a model provider

1. Implement `ModelAdapter.generate(messages, tools)`.
2. Normalize text, tool requests, and optional token counts into `ModelResponse`.
3. Convert provider exceptions to `ModelError`.
4. Add the provider to `ModelConfig` and `models/factory.py`.

Provider SDKs are optional extras so mock-based research and tests do not install paid-provider clients.

## Batch experiments

Edit `agent_research/experiments/configs/baseline.yaml` and `datasets/tasks.json`, then run:

```powershell
agent-experiment agent_research/experiments/configs/baseline.yaml
```

The YAML supports multiple model configurations, memory enablement and `top_k`, `max_steps`, task/trace/result paths, and a reproducibility seed. The runner creates one JSON trace per run and a CSV containing success, steps, tool-call count, latency, available token usage, answer, and error. `expected_contains` in a task provides a deliberately simple optional evaluation.

Example configuration:

```yaml
models:
  - provider: openai
    model: gpt-5.6
    max_output_tokens: 1024
  - provider: anthropic
    model: claude-sonnet-4-20250514
memory:
  enabled: true
  top_k: 5
  path: ../../../memory.sqlite3
max_steps: 8
tasks_file: ../../../datasets/tasks.json
trace_dir: ../../../traces
results_file: ../../../results/comparison.csv
seed: 42
```

## Tests

```powershell
pytest
```

The suite covers the registry, tools, persistent retrieval, state transitions, max-step behavior, trace JSON, configuration loading, and an integration path where the mock requests the calculator, receives `42`, and returns a final answer without an API call.
