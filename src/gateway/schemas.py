"""OpenAI-compatible request bodies, plus the gateway's own fields."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: str
    content: str | list[Any] | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    """Subset of the OpenAI chat completions request, plus an optional policy."""

    model_config = ConfigDict(extra="ignore")

    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = 0
    max_tokens: int | None = None
    stream: bool = False
    policy: str | None = None


class KeyCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1, max_length=80)
    quota_tokens: int = Field(default=1_000_000, gt=0)


def flatten_content(content: str | list[Any] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type", "text") == "text":
            parts.append(str(item.get("text", "")))
    return "\n".join(part for part in parts if part)
