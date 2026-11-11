"""Local Ollama chat API. No key; the base URL is the only configuration."""

import httpx

from gateway.catalog import ModelSpec
from gateway.errors import ProviderError
from gateway.providers.base import BaseProvider, ChatResult
from gateway.textutil import estimate_tokens, message_text


class OllamaProvider(BaseProvider):
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    async def complete(
        self,
        model: ModelSpec,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        try:
            response = await self._client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": model.upstream_id,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
        except httpx.TimeoutException as exc:
            raise ProviderError("ollama timed out", 504, True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError("ollama connection error", 502, True) from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise ProviderError(
                f"ollama returned {response.status_code}", response.status_code, True
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"ollama returned {response.status_code}",
                response.status_code,
                False,
            )
        data = response.json()
        message = data.get("message") or {}
        content = str(message.get("content") or "")
        prompt_tokens = int(
            data.get("prompt_eval_count") or estimate_tokens(message_text(messages))
        )
        completion_tokens = int(data.get("eval_count") or estimate_tokens(content))
        return ChatResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model.id,
            provider=self.name,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
