"""Construct adapters from configuration."""

from agent_research.core.config import ModelConfig
from agent_research.models.anthropic_adapter import AnthropicAdapter
from agent_research.models.base import ModelAdapter
from agent_research.models.gemini_adapter import GeminiAdapter
from agent_research.models.mock_adapter import MockModelAdapter
from agent_research.models.openai_adapter import OpenAIAdapter


def create_model(config: ModelConfig) -> ModelAdapter:
    adapters = {
        "openai": OpenAIAdapter,
        "anthropic": AnthropicAdapter,
        "gemini": GeminiAdapter,
    }
    if config.provider == "mock":
        return MockModelAdapter(model_name=config.model)
    return adapters[config.provider](
        config.model, config.temperature, config.max_output_tokens
    )
