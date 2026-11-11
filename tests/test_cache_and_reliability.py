"""Cache, embeddings, token bucket, breaker, and failover."""

import asyncio
import time

import fakeredis
import pytest

from gateway.cache.embeddings import HashEmbedder, build_embedder, cosine, tokenize
from gateway.cache.semantic import SemanticCache
from gateway.catalog import ModelSpec
from gateway.errors import ProviderError
from gateway.providers.base import BaseProvider, ChatResult
from gateway.providers.registry import Registry
from gateway.reliability.breaker import CircuitBreaker
from gateway.reliability.bucket import TokenBucket
from gateway.reliability.executor import Executor


def test_hash_embeddings_rank_paraphrases_above_unrelated_text() -> None:
    embedder = HashEmbedder()
    france = embedder.embed("What is the capital of France?")
    paris = embedder.embed("What is France's capital?")
    noun = embedder.embed("What is a noun?")
    assert cosine(france, paris) >= 0.80
    assert cosine(france, noun) < 0.80
    assert tokenize("What is a noun?") == ["noun"]
    assert embedder.embed("") == [0.0] * embedder.dim
    with pytest.raises(ValueError):
        HashEmbedder(4)
    with pytest.raises(ValueError):
        build_embedder("bow")
    with pytest.raises(RuntimeError):
        build_embedder("sentence-transformers")


def test_memory_cache_exact_semantic_ttl_and_tenant() -> None:
    cache = SemanticCache(HashEmbedder(), ttl_seconds=60, threshold=0.80, max_entries=2)
    payload = {"content": "paris", "model": "mock-small"}
    cache.set(
        exact_key="k1",
        text="What is the capital of France?",
        tenant="1",
        policy="cost",
        temperature=0,
        max_tokens=256,
        payload=payload,
    )
    exact = cache.get(
        exact_key="k1",
        text="What is the capital of France?",
        tenant="1",
        policy="cost",
        temperature=0,
        max_tokens=256,
    )
    assert exact is not None and exact.kind == "exact"
    semantic = cache.get(
        exact_key="other",
        text="What is France's capital?",
        tenant="1",
        policy="cost",
        temperature=0,
        max_tokens=256,
    )
    assert semantic is not None and semantic.kind == "semantic"
    other_tenant = cache.get(
        exact_key="other",
        text="What is France's capital?",
        tenant="2",
        policy="cost",
        temperature=0,
        max_tokens=256,
    )
    assert other_tenant is None
    cache._entries[0].expires_at = time.time() - 5
    expired = cache.get(
        exact_key="k1",
        text="What is the capital of France?",
        tenant="1",
        policy="cost",
        temperature=0,
        max_tokens=256,
    )
    assert expired is None

    bounded = SemanticCache(HashEmbedder(), ttl_seconds=60, threshold=0.99, max_entries=1)
    for index in range(3):
        bounded.set(
            exact_key=f"k{index}",
            text=f"unique prompt number {index} about zebras",
            tenant="1",
            policy="cost",
            temperature=0,
            max_tokens=32,
            payload={"content": str(index)},
        )
    assert len(bounded._entries) == 1


def test_redis_cache_roundtrip_and_fallback() -> None:
    client = fakeredis.FakeRedis()
    writer = SemanticCache(
        HashEmbedder(),
        ttl_seconds=60,
        threshold=0.8,
        redis_client=client,
    )
    writer.set(
        exact_key="abc",
        text="Define photosynthesis.",
        tenant="9",
        policy="cost",
        temperature=0,
        max_tokens=128,
        payload={"content": "plants"},
    )
    reader = SemanticCache(
        HashEmbedder(),
        ttl_seconds=60,
        threshold=0.8,
        redis_client=client,
    )
    hit = reader.get(
        exact_key="abc",
        text="Define photosynthesis.",
        tenant="9",
        policy="cost",
        temperature=0,
        max_tokens=128,
    )
    assert hit is not None and hit.payload["content"] == "plants"
    assert reader.backend == "redis"

    class Boom:
        def get(self, _key: str) -> bytes:
            raise RuntimeError("redis down")

        def set(self, _key: str, _value: str) -> None:
            raise RuntimeError("redis down")

    flaky = SemanticCache(HashEmbedder(), ttl_seconds=30, threshold=0.8, redis_client=Boom())
    assert flaky.backend == "redis"
    flaky.set(
        exact_key="z",
        text="hello there",
        tenant="1",
        policy="cost",
        temperature=0,
        max_tokens=16,
        payload={"content": "hi"},
    )
    assert flaky.backend == "memory"
    assert flaky.get(
        exact_key="z",
        text="hello there",
        tenant="1",
        policy="cost",
        temperature=0,
        max_tokens=16,
    )

    offline = SemanticCache(
        HashEmbedder(),
        ttl_seconds=10,
        threshold=0.8,
        redis_url="redis://127.0.0.1:1/0",
        connect_retries=1,
    )
    assert offline.backend == "memory"


def test_bucket_and_breaker() -> None:
    bucket = TokenBucket(rate=10, capacity=1)
    assert bucket.consume()
    assert not bucket.consume()
    bucket.updated -= 5
    assert bucket.consume()
    with pytest.raises(ValueError):
        TokenBucket(rate=-1, capacity=1)

    breaker = CircuitBreaker(failure_threshold=2, reset_seconds=10)
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state == "closed"
    breaker.record_failure()
    assert breaker.state == "open"
    assert not breaker.allow()
    breaker.opened_at -= 11
    assert breaker.allow()
    assert breaker.state == "half_open"
    assert not breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open"
    breaker.opened_at -= 11
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed"


class _Scripted(BaseProvider):
    def __init__(self, name: str, script: list[object]) -> None:
        self.name = name
        self.script = list(script)
        self.calls = 0

    async def complete(
        self,
        model: ModelSpec,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> ChatResult:
        del messages, temperature, max_tokens
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        assert isinstance(item, ChatResult)
        return item


def _spec(model_id: str, provider: str, tier: int = 0) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        provider=provider,
        upstream_id=model_id,
        tier=tier,
        input_cost_per_million=1,
        output_cost_per_million=1,
        latency_rank=1,
        quality_rank=1,
    )


def _result(model_id: str, provider: str) -> ChatResult:
    return ChatResult("ok", 1, 1, model_id, provider)


def test_executor_retries_and_fails_over() -> None:
    primary = _Scripted(
        "a",
        [ProviderError("nope", 503, True), _result("a-small", "a")],
    )
    backup = _Scripted("b", [_result("b-small", "b")])
    registry = Registry(
        {"a": primary, "b": backup},
        [_spec("a-small", "a"), _spec("b-small", "b")],
    )
    executor = Executor(
        registry,
        max_retries=2,
        backoff_seconds=0,
        failure_threshold=5,
        reset_seconds=30,
        rate=10,
        capacity=10,
    )
    result, failovers = asyncio.run(
        executor.execute(
            "a-small", [{"role": "user", "content": "hi"}], temperature=0, max_tokens=8
        )
    )
    assert result.content == "ok"
    assert failovers == 0
    assert primary.calls == 2

    primary.script = [ProviderError("still", 503, True)] * 3
    result, failovers = asyncio.run(
        executor.execute(
            "a-small", [{"role": "user", "content": "hi"}], temperature=0, max_tokens=8
        )
    )
    assert result.provider == "b"
    assert failovers == 1

    with pytest.raises(ProviderError):
        asyncio.run(executor.execute("missing", [], temperature=0, max_tokens=8))


def test_executor_skips_open_circuits_and_empty_buckets() -> None:
    primary = _Scripted("a", [ProviderError("should not run", 500, True)])
    backup = _Scripted("b", [_result("b-small", "b")])
    registry = Registry(
        {"a": primary, "b": backup},
        [_spec("a-small", "a"), _spec("b-small", "b")],
    )
    executor = Executor(
        registry,
        max_retries=0,
        backoff_seconds=0,
        failure_threshold=1,
        reset_seconds=60,
        rate=0,
        capacity=1,
    )
    executor.breakers["a"].state = "open"
    executor.breakers["a"].opened_at = time.monotonic()
    result, failovers = asyncio.run(
        executor.execute(
            "a-small", [{"role": "user", "content": "hi"}], temperature=0, max_tokens=4
        )
    )
    assert result.provider == "b"
    assert failovers == 1
    assert primary.calls == 0

    executor.breakers["a"].state = "closed"
    executor.buckets["a"].tokens = 0
    executor.buckets["a"].rate = 0
    executor.buckets["b"].tokens = 1
    backup.script = [_result("b-small", "b")]
    result, failovers = asyncio.run(
        executor.execute(
            "a-small", [{"role": "user", "content": "hi"}], temperature=0, max_tokens=4
        )
    )
    assert result.provider == "b"
    assert primary.calls == 0


def test_executor_does_not_failover_on_client_errors() -> None:
    primary = _Scripted("a", [ProviderError("bad request", 400, False)])
    backup = _Scripted("b", [_result("b-small", "b")])
    registry = Registry(
        {"a": primary, "b": backup},
        [_spec("a-small", "a"), _spec("b-small", "b", tier=1)],
    )
    executor = Executor(
        registry,
        max_retries=1,
        backoff_seconds=0,
        failure_threshold=3,
        reset_seconds=30,
        rate=5,
        capacity=5,
    )
    with pytest.raises(ProviderError) as caught:
        asyncio.run(
            executor.execute(
                "a-small",
                [{"role": "user", "content": "hi"}],
                temperature=0,
                max_tokens=4,
            )
        )
    assert caught.value.status_code == 400
    assert backup.calls == 0
