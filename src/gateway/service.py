"""Chat completion pipeline: guardrails, cache, route, record."""

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from gateway.cache.semantic import SemanticCache
from gateway.catalog import ModelSpec
from gateway.config import Settings
from gateway.errors import GatewayError
from gateway.guardrails.budget import assert_budget
from gateway.guardrails.hooks import GuardContext, HookManager
from gateway.logging_config import log_extra
from gateway.observability.metrics import GatewayMetrics
from gateway.observability.tracing import get_tracer
from gateway.providers.registry import Registry
from gateway.reliability.executor import Executor
from gateway.routing.graph import RouterGraph
from gateway.schemas import ChatRequest, flatten_content
from gateway.tenancy.store import KeyRecord, Store
from gateway.textutil import canonical_messages, estimate_tokens

logger = logging.getLogger(__name__)
_AUTO = {"auto", "gateway-auto"}
_POLICIES = {"cost", "latency", "quality"}
_ROLES = {"system", "user", "assistant", "tool", "developer"}


@dataclass
class Completion:
    id: str
    created: int
    model: str
    initial_model: str
    provider: str
    content: str
    prompt_tokens: int
    completion_tokens: int
    complexity: str
    classifier: str
    policy: str
    cached: bool
    cache_kind: str
    escalated: bool
    failover_count: int
    quality_ok: bool
    verify_reason: str
    cost_usd: float
    cost_saved_usd: float
    latency_ms: float


class ChatService:
    def __init__(
        self,
        *,
        settings: Settings,
        store: Store,
        registry: Registry,
        cache: SemanticCache,
        executor: Executor,
        graph: RouterGraph,
        hooks: HookManager,
        metrics: GatewayMetrics,
    ) -> None:
        self.settings = settings
        self.store = store
        self.registry = registry
        self.cache = cache
        self.executor = executor
        self.graph = graph
        self.hooks = hooks
        self.metrics = metrics

    async def complete(
        self,
        request: ChatRequest,
        key: KeyRecord,
        *,
        policy_header: str | None,
    ) -> Completion:
        started = time.perf_counter()
        try:
            completion = await self._complete(
                request, key, policy_header=policy_header, started=started
            )
        except GatewayError as exc:
            self.metrics.observe_error(exc.code, time.perf_counter() - started)
            logger.warning(
                "chat failed",
                extra=log_extra(code=exc.code, status=exc.status_code, key_prefix=key.key_prefix),
            )
            raise
        except Exception:
            self.metrics.observe_error("internal", time.perf_counter() - started)
            logger.exception("chat crashed")
            raise GatewayError("Internal error", 500, "server_error", "internal") from None
        self.metrics.observe_success(
            policy=completion.policy,
            complexity=completion.complexity,
            cached=completion.cached,
            cache_kind=completion.cache_kind,
            provider=completion.provider,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            cost_usd=completion.cost_usd,
            cost_saved_usd=completion.cost_saved_usd,
            failover_count=completion.failover_count,
            escalated=completion.escalated,
            latency_seconds=completion.latency_ms / 1000,
        )
        logger.info(
            "chat completed",
            extra=log_extra(
                model=completion.model,
                provider=completion.provider,
                complexity=completion.complexity,
                policy=completion.policy,
                cached=completion.cached,
                cache_kind=completion.cache_kind,
                escalated=completion.escalated,
                latency_ms=round(completion.latency_ms, 2),
                key_prefix=key.key_prefix,
            ),
        )
        return completion

    async def _complete(
        self,
        request: ChatRequest,
        key: KeyRecord,
        *,
        policy_header: str | None,
        started: float,
    ) -> Completion:
        messages = normalize_messages(request)
        temperature = 0.0 if request.temperature is None else float(request.temperature)
        if not 0.0 <= temperature <= 2.0:
            raise GatewayError(
                "temperature must be between 0 and 2",
                400,
                "invalid_request_error",
                "temperature",
            )
        max_tokens = request.max_tokens or self.settings.default_max_tokens
        policy, requested = resolve_policy(
            request.model,
            request.policy or policy_header or self.settings.default_policy,
            self.registry,
        )
        prompt_tokens = estimate_tokens("\n".join(message["content"] for message in messages))
        assert_budget(
            max_tokens=max_tokens,
            prompt_tokens=prompt_tokens,
            key=key,
            settings=self.settings,
        )
        ctx = GuardContext(messages=messages, model=request.model, max_tokens=max_tokens)
        ctx = await self.hooks.run_pre(ctx)
        canonical = canonical_messages(ctx.messages)
        exact_key = _exact_key(
            tenant=str(key.id),
            policy=policy,
            temperature=temperature,
            max_tokens=max_tokens,
            canonical=canonical,
        )
        tracer = get_tracer()
        with tracer.start_as_current_span("gateway.chat") as span:
            span.set_attribute("gateway.policy", policy)
            span.set_attribute("gateway.tenant", key.key_prefix)
            hit = self.cache.get(
                exact_key=exact_key,
                text=canonical,
                tenant=str(key.id),
                policy=policy,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if hit is not None:
                span.set_attribute("gateway.cache", hit.kind)
                completion = self._from_cache(hit.payload, cache_kind=hit.kind, policy=policy)
            else:
                span.set_attribute("gateway.cache", "miss")
                completion = await self._generate(
                    ctx.messages,
                    policy=policy,
                    requested=requested,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                completion.content = await self.hooks.run_post(completion.content, ctx)
                self.cache.set(
                    exact_key=exact_key,
                    text=canonical,
                    tenant=str(key.id),
                    policy=policy,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    payload=_cache_payload(completion),
                )
        completion.latency_ms = (time.perf_counter() - started) * 1000
        self.store.add_usage(
            key.id,
            model_id=completion.model,
            provider=completion.provider,
            complexity=completion.complexity,
            policy=completion.policy,
            prompt_tokens=completion.prompt_tokens,
            completion_tokens=completion.completion_tokens,
            cached=completion.cached,
            escalated=completion.escalated,
            failover_count=completion.failover_count,
            cost_usd=completion.cost_usd,
            cost_saved_usd=completion.cost_saved_usd,
            latency_ms=completion.latency_ms,
            charge_tokens=0
            if completion.cached
            else completion.prompt_tokens + completion.completion_tokens,
        )
        return completion

    def _from_cache(self, payload: dict[str, Any], *, cache_kind: str, policy: str) -> Completion:
        prompt_tokens = int(payload["prompt_tokens"])
        completion_tokens = int(payload["completion_tokens"])
        model = self.registry.get(str(payload["model"]))
        _, saved = price(
            model, self.registry.biggest(), prompt_tokens, completion_tokens, cached=True
        )
        return Completion(
            id=_new_id(),
            created=_created(),
            model=str(payload["model"]),
            initial_model=str(payload.get("initial_model") or payload["model"]),
            provider=str(payload["provider"]),
            content=str(payload["content"]),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            complexity=str(payload["complexity"]),
            classifier=str(payload.get("classifier") or "cache"),
            policy=policy,
            cached=True,
            cache_kind=cache_kind,
            escalated=False,
            failover_count=0,
            quality_ok=bool(payload.get("quality_ok", True)),
            verify_reason=str(payload.get("verify_reason") or "cache"),
            cost_usd=0.0,
            cost_saved_usd=saved,
            latency_ms=0.0,
        )

    async def _generate(
        self,
        messages: list[dict[str, str]],
        *,
        policy: str,
        requested: str | None,
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        state = await self.graph.run(
            messages=messages,
            policy=policy,
            requested_model=requested,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if state.get("error"):
            status = int(state.get("error_status") or 502)
            raise GatewayError(
                str(state.get("error") or "upstream failed"),
                status,
                "server_error" if status >= 500 else "invalid_request_error",
                "upstream",
            )
        model_id = str(state.get("model_id") or "")
        model = self.registry.get(model_id)
        prompt_tokens = int(state.get("prompt_tokens") or 0)
        completion_tokens = int(state.get("completion_tokens") or 0)
        actual, saved = price(
            model, self.registry.biggest(), prompt_tokens, completion_tokens, cached=False
        )
        return Completion(
            id=_new_id(),
            created=_created(),
            model=model_id,
            initial_model=str(state.get("initial_model") or model_id),
            provider=str(state.get("provider") or (model.provider if model else "")),
            content=str(state.get("content") or ""),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            complexity=str(state.get("complexity") or "medium"),
            classifier=str(state.get("classifier") or "heuristic"),
            policy=policy,
            cached=False,
            cache_kind="miss",
            escalated=bool(state.get("escalated")),
            failover_count=int(state.get("failover_count") or 0),
            quality_ok=bool(state.get("verified")),
            verify_reason=str(state.get("verify_reason") or ""),
            cost_usd=actual,
            cost_saved_usd=saved,
            latency_ms=0.0,
        )


def normalize_messages(request: ChatRequest) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for message in request.messages:
        role = message.role.strip().lower()
        if role not in _ROLES:
            raise GatewayError(
                f"Unsupported message role '{message.role}'",
                400,
                "invalid_request_error",
                "role",
            )
        if role == "developer":
            role = "system"
        messages.append({"role": role, "content": flatten_content(message.content)})
    if not any(message["content"].strip() for message in messages):
        raise GatewayError("Messages are empty", 400, "invalid_request_error", "messages")
    return messages


def resolve_policy(model: str, requested_policy: str, registry: Registry) -> tuple[str, str | None]:
    if model in _AUTO:
        policy = requested_policy
        if policy not in _POLICIES:
            raise GatewayError(
                "policy must be cost, latency, or quality",
                400,
                "invalid_request_error",
                "policy",
            )
        return policy, None
    if registry.get(model) is None:
        raise GatewayError(
            f"Model '{model}' is not available",
            404,
            "invalid_request_error",
            "model_not_found",
        )
    return "explicit", model


def price(
    model: ModelSpec | None,
    biggest: ModelSpec,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cached: bool,
) -> tuple[float, float]:
    baseline = biggest.cost(prompt_tokens, completion_tokens)
    actual = 0.0 if cached or model is None else model.cost(prompt_tokens, completion_tokens)
    return actual, max(0.0, baseline - actual)


def completion_payload(completion: Completion) -> dict[str, Any]:
    return {
        "id": completion.id,
        "object": "chat.completion",
        "created": completion.created,
        "model": completion.model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": completion.content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "total_tokens": completion.prompt_tokens + completion.completion_tokens,
        },
        "gateway": gateway_metadata(completion),
    }


def gateway_metadata(completion: Completion) -> dict[str, Any]:
    return {
        "provider": completion.provider,
        "complexity": completion.complexity,
        "classifier": completion.classifier,
        "policy": completion.policy,
        "cached": completion.cached,
        "cache_kind": completion.cache_kind,
        "escalated": completion.escalated,
        "initial_model": completion.initial_model,
        "failover_count": completion.failover_count,
        "quality_ok": completion.quality_ok,
        "verify_reason": completion.verify_reason,
        "cost_usd": completion.cost_usd,
        "cost_saved_usd": completion.cost_saved_usd,
        "latency_ms": round(completion.latency_ms, 3),
    }


def response_headers(completion: Completion) -> dict[str, str]:
    return {
        "x-gateway-model": completion.model,
        "x-gateway-cache": completion.cache_kind,
        "x-gateway-complexity": completion.complexity,
        "x-gateway-escalated": "true" if completion.escalated else "false",
    }


def iter_sse(completion: Completion) -> Any:
    created = completion.created
    base = {
        "id": completion.id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": completion.model,
    }
    yield _sse(
        {
            **base,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
    )
    for piece in _chunks(completion.content):
        yield _sse(
            {
                **base,
                "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
            }
        )
    yield _sse(
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": completion.prompt_tokens,
                "completion_tokens": completion.completion_tokens,
                "total_tokens": completion.prompt_tokens + completion.completion_tokens,
            },
            "gateway": gateway_metadata(completion),
        }
    )
    yield "data: [DONE]\n\n"


def _chunks(text: str, size: int = 24) -> list[str]:
    if not text:
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def _exact_key(
    *, tenant: str, policy: str, temperature: float, max_tokens: int, canonical: str
) -> str:
    raw = f"{tenant}|{policy}|{temperature:.3f}|{max_tokens}|{canonical}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_payload(completion: Completion) -> dict[str, Any]:
    return {
        "content": completion.content,
        "model": completion.model,
        "initial_model": completion.initial_model,
        "provider": completion.provider,
        "prompt_tokens": completion.prompt_tokens,
        "completion_tokens": completion.completion_tokens,
        "complexity": completion.complexity,
        "classifier": completion.classifier,
        "escalated": completion.escalated,
        "failover_count": completion.failover_count,
        "quality_ok": completion.quality_ok,
        "verify_reason": completion.verify_reason,
    }


def _new_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:24]


def _created() -> int:
    return int(time.time())
