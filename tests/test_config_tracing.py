import json

from agent_research.core.config import ExperimentConfig, ModelConfig
from agent_research.core.tracing import JsonTracer


def test_experiment_config_loading_resolves_paths(tmp_path) -> None:
    config_file = tmp_path / "experiment.yaml"
    config_file.write_text("models: [mock]\ntasks_file: tasks.json\nmemory:\n  enabled: false\n", encoding="utf-8")
    config = ExperimentConfig.from_yaml(config_file)
    assert config.models[0].provider == "mock"
    assert config.tasks_file == (tmp_path / "tasks.json").resolve()


def test_model_config_migrates_legacy_max_tokens_name() -> None:
    config = ModelConfig.model_validate(
        {
            "provider": "openai",
            "model": "gpt-5.6",
            "max_tokens": 2048,
        }
    )

    assert config.max_output_tokens == 2048
    assert "max_tokens" not in config.model_dump()


def test_trace_generation(tmp_path) -> None:
    tracer = JsonTracer(tmp_path, "run-1", "task", "mock", "model")
    tracer.record("model_response", 1, model_response={"content": "done"})
    path = tracer.finish("done", [], 2, 1)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == "1.0"
    assert data["events"][0]["run_id"] == "run-1"
    assert data["final_answer"] == "done"
