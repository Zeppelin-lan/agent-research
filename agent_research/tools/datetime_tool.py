"""Timezone-aware current datetime tool."""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel

from agent_research.tools.registry import ToolError


class DateTimeInput(BaseModel):
    timezone: str = "UTC"


def current_datetime(inputs: DateTimeInput) -> str:
    try:
        return datetime.now(ZoneInfo(inputs.timezone)).isoformat()
    except ZoneInfoNotFoundError as exc:
        raise ToolError(f"Unknown timezone: {inputs.timezone}") from exc
