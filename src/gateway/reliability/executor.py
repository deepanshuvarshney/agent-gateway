"""Call a model with retries, then fail over to another provider."""

import asyncio
import logging

from gateway.catalog import ModelSpec
from gateway.errors import ProviderError
from gateway.providers.base import ChatResult
from gateway.providers.registry import Registry
from gateway.reliability.breaker import CircuitBreaker
from gateway.reliability.bucket import TokenBucket

logger = logging.getLogger(__name__)


class Executor:
    def __init__(
        self,
        registry: Registry,
        *,
        max_retries: int,
        backoff_seconds: float,
        failure_threshold: int,
        reset_seconds: float,
        rate: float,
        capacity: float,
    ) -> None:
        self.registry = registry
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.breakers = {
            name: CircuitBreaker(failure_threshold, reset_seconds) for name in registry.providers
        }
        self.buckets = {name: TokenBucket(rate, capacity) for name in registry.providers}

    async def execute(
        self,
        model_id: str,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> tuple[ChatResult, int]:
        chain = self.registry.failover_chain(model_id)
        if not chain:
            raise ProviderError(f"Unknown model '{model_id}'", 404, False)
        failovers = 0
        last_error: ProviderError | None = None
        for model in chain:
            result, switches, last_error, done = await self._attempt(
                model, messages, temperature, max_tokens
            )
            failovers += switches
            if done and result is not None:
                return result, failovers
        if last_error is None:
            last_error = ProviderError("No provider available", 503, True)
        raise last_error

    async def _attempt(
        self,
        model: ModelSpec,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[ChatResult | None, int, ProviderError | None, bool]:
        provider = self.registry.provider(model.provider)
        breaker = self.breakers[provider.name]
        bucket = self.buckets[provider.name]
        if not breaker.allow():
            logger.info("circuit open provider=%s", provider.name)
            return None, 1, ProviderError(f"{provider.name} circuit open", 503, True), False
        if not bucket.consume():
            logger.info("rate limited provider=%s", provider.name)
            return None, 1, ProviderError(f"{provider.name} rate limited", 429, True), False
        last_error: ProviderError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                result = await provider.complete(
                    model,
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                breaker.record_success()
                return result, 0, None, True
            except ProviderError as exc:
                breaker.record_failure()
                last_error = exc
                if not exc.retryable:
                    raise
                if attempt < self.max_retries:
                    delay = self.backoff_seconds * (2**attempt)
                    if delay > 0:
                        await asyncio.sleep(delay)
                    continue
        return None, 1, last_error, False
