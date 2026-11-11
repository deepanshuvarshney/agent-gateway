"""Deterministic provider used by tests, CI, and the free deploy.

Smaller models return shallow answers. The verifier escalates when the prompt
actually needed more depth, which is how the router/verifier loop is exercised
without a live model.
"""

import asyncio
import hashlib

from gateway.catalog import ModelSpec
from gateway.errors import ProviderError
from gateway.providers.base import BaseProvider, ChatResult
from gateway.textutil import estimate_tokens, message_text

_MEDIUM = (
    "Here is a concise explanation of the question, with one concrete example "
    "and a practical takeaway you can apply immediately."
)
_LARGE = (
    "Detailed answer. Because the prompt needs a careful argument, this response "
    "walks through the reasoning, the edge cases, and the trade-off between the "
    "main approaches. The conclusion follows from those constraints, and a small "
    "example shows how to check the result before relying on it."
)


class MockProvider(BaseProvider):
    name = "mock"

    def __init__(self, latency_ms: int = 0) -> None:
        self.latency_ms = latency_ms
        self.fail_next = 0
        self.fail_status = 503
        self.calls = 0
        self.last_messages: list[dict[str, str]] = []

    async def complete(
        self,
        model: ModelSpec,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        del temperature, max_tokens
        self.calls += 1
        self.last_messages = [dict(message) for message in messages]
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms * (1 + model.tier) / 1000)
        if self.fail_next > 0:
            self.fail_next -= 1
            retryable = self.fail_status == 429 or self.fail_status >= 500
            raise ProviderError("mock injected failure", self.fail_status, retryable)
        prompt = message_text(messages)
        if "[[fail-provider]]" in prompt:
            raise ProviderError("mock prompt failure", 503, True)
        content = render_answer(model, prompt)
        return ChatResult(
            content=content,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(content),
            model=model.id,
            provider=self.name,
        )


def render_answer(model: ModelSpec, prompt: str) -> str:
    digest = hashlib.sha256(f"{model.id}:{prompt}".encode()).hexdigest()[:8]
    if "[[LOW_QUALITY]]" in prompt:
        return "[[LOW_QUALITY]] too thin."
    if model.tier <= 0:
        return f"Short reply {digest}."
    if model.tier == 1:
        return f"Ref {digest}. {_MEDIUM}"
    return f"Ref {digest}. {_LARGE}"
