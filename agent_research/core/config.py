"""Typed runtime and experiment configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelConfig(BaseModel):
    provider: Literal["openai", "anthropic", "gemini", "mock"] = "mock"
    model: str = "mock-model"
    temperature: float | None = None
    max_output_tokens: int = Field(default=1024, gt=0)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_token_name(cls, value: Any) -> Any:
        """Accept old experiment YAML while emitting only the current name."""
        if isinstance(value, dict) and "max_tokens" in value:
            data = dict(value)
            data.setdefault("max_output_tokens", data.pop("max_tokens"))
            return data
        return value


class MemoryConfig(BaseModel):
    enabled: bool = True
    top_k: int = Field(default=5, ge=1)
    path: Path = Path("memory.sqlite3")


class RuntimeConfig(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    max_steps: int = Field(default=8, ge=1)
    trace_dir: Path = Path("traces")
    seed: int = 0


class ExperimentConfig(BaseModel):
    models: list[ModelConfig]
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    max_steps: int = Field(default=8, ge=1)
    tasks_file: Path
    trace_dir: Path = Path("traces")
    results_file: Path = Path("results/results.csv")
    seed: int = 0

    @model_validator(mode="before")
    @classmethod
    def expand_model_names(cls, value: Any) -> Any:
        if isinstance(value, dict):
            data = dict(value)
            data["models"] = [
                {"provider": item, "model": f"{item}-default"}
                if isinstance(item, str)
                else item
                for item in data.get("models", [])
            ]
            return data
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        config_path = Path(path)
        with config_path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        config = cls.model_validate(raw)
        base = config_path.parent
        for field in ("tasks_file", "trace_dir", "results_file"):
            current = getattr(config, field)
            if not current.is_absolute():
                setattr(config, field, (base / current).resolve())
        if not config.memory.path.is_absolute():
            config.memory.path = (base / config.memory.path).resolve()
        return config
