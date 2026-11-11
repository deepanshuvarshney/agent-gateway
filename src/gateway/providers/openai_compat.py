"""Groq and Gemini both expose an OpenAI-compatible chat completions endpoint."""

import httpx

from gateway.catalog import ModelSpec
from gateway.errors import ProviderError
from gateway.providers.base import BaseProvider, ChatResult
from gateway.textutil import estimate_tokens


class OpenAICompatProvider(BaseProvider):
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
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
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": model.upstream_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(f"{self.name} timed out", 504, True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"{self.name} connection error", 502, True) from exc
        if response.status_code == 429 or response.status_code >= 500:
            raise ProviderError(
                f"{self.name} returned {response.status_code}",
                response.status_code,
                True,
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name} returned {response.status_code}",
                response.status_code,
                False,
            )
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.name} returned no choices", 502, True)
        message = choices[0].get("message") or {}
        content = str(message.get("content") or "")
        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or estimate_tokens(content))
        completion_tokens = int(usage.get("completion_tokens") or estimate_tokens(content))
        finish = str(choices[0].get("finish_reason") or "stop")
        return ChatResult(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model=model.id,
            provider=self.name,
            finish_reason=finish,
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
