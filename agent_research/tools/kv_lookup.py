"""Key-value lookup backed by inline data or a JSON file."""

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, model_validator

from agent_research.tools.registry import ToolError


class KVLookupInput(BaseModel):
    key: str
    data: dict[str, Any] | None = None
    file: Path | None = None

    @model_validator(mode="after")
    def require_source(self) -> "KVLookupInput":
        if self.data is None and self.file is None:
            raise ValueError("Provide data or file")
        return self


def kv_lookup(inputs: KVLookupInput) -> Any:
    data = inputs.data
    if data is None and inputs.file is not None:
        try:
            loaded = json.loads(inputs.file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ToolError(f"Cannot read JSON data: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ToolError("JSON root must be an object")
        data = loaded
    assert data is not None
    if inputs.key not in data:
        raise ToolError(f"Key not found: {inputs.key}")
    return data[inputs.key]
