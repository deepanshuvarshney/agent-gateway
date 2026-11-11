"""Provider interface."""

from dataclasses import dataclass

from gateway.catalog import ModelSpec


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    provider: str
    finish_reason: str = "stop"


class BaseProvider:
    name: str

    async def complete(
        self,
        model: ModelSpec,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        return None
