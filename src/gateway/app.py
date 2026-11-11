"""FastAPI application factory."""

import logging
import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from gateway import __version__
from gateway.cache.embeddings import build_embedder
from gateway.cache.semantic import SemanticCache
from gateway.config import Settings
from gateway.dashboard import load_html
from gateway.errors import GatewayError
from gateway.guardrails.hooks import HookManager
from gateway.logging_config import configure_logging
from gateway.observability.metrics import GatewayMetrics, metrics_content_type
from gateway.observability.tracing import setup_tracing
from gateway.providers.registry import build_registry
from gateway.reliability.executor import Executor
from gateway.routing.graph import RouterGraph
from gateway.schemas import ChatRequest, KeyCreate
from gateway.service import ChatService, completion_payload, iter_sse, response_headers
from gateway.tenancy.store import KeyRecord, Store, hash_key

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    if settings.env == "production" and settings.admin_api_key == "gw_admin_dev":
        raise RuntimeError(
            "GATEWAY_ADMIN_API_KEY is still the local default. "
            "Set a unique admin key before running with GATEWAY_ENV=production."
        )
    configure_logging(settings.log_level)
    setup_tracing(settings)
    store = Store(settings.database_url)
    store.ensure_key(
        settings.admin_api_key,
        name="admin",
        quota_tokens=10**12,
        is_admin=True,
    )
    store.ensure_key(
        settings.bootstrap_api_key,
        name="demo",
        quota_tokens=settings.default_quota_tokens,
        is_admin=False,
    )
    embedder = build_embedder(settings.embeddings, settings.embedding_dim)
    cache = SemanticCache(
        embedder,
        ttl_seconds=settings.cache_ttl_seconds,
        threshold=settings.cache_similarity_threshold,
        max_entries=settings.cache_max_entries,
        redis_url=settings.redis_url,
        connect_retries=settings.redis_connect_retries,
    )
    registry = build_registry(settings)
    executor = Executor(
        registry,
        max_retries=settings.max_retries,
        backoff_seconds=settings.retry_backoff_seconds,
        failure_threshold=settings.circuit_failure_threshold,
        reset_seconds=settings.circuit_reset_seconds,
        rate=settings.provider_rps,
        capacity=settings.provider_burst,
    )
    graph = RouterGraph(
        registry,
        executor,
        embedder,
        allow_escalation=settings.enable_escalation,
    )
    hooks = HookManager(
        max_prompt_chars=settings.max_prompt_chars,
        injection_mode=settings.injection_mode,
    )
    metrics = GatewayMetrics()
    service = ChatService(
        settings=settings,
        store=store,
        registry=registry,
        cache=cache,
        executor=executor,
        graph=graph,
        hooks=hooks,
        metrics=metrics,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "gateway started",
            extra={
                "extra_fields": {
                    "version": __version__,
                    "providers": sorted(registry.providers),
                    "cache": cache.backend,
                    "embeddings": settings.embeddings,
                }
            },
        )
        yield
        await registry.aclose()

    app = FastAPI(title="Agent Gateway", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.cache = cache
    app.state.registry = registry
    app.state.service = service
    app.state.metrics = metrics
    app.state.hooks = hooks

    @app.exception_handler(GatewayError)
    async def on_gateway_error(_request: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_body())

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        message = "Invalid request"
        errors = exc.errors()
        if errors:
            location = ".".join(str(part) for part in errors[0].get("loc", []))
            message = f"{location}: {errors[0].get('msg', 'invalid')}"
        return JSONResponse(
            status_code=400,
            content={
                "error": {"message": message, "type": "invalid_request_error", "code": "validation"}
            },
        )

    @app.get("/health")
    def health() -> JSONResponse:
        try:
            store.ping()
        except Exception:
            logger.exception("health check failed")
            return JSONResponse(status_code=503, content={"status": "degraded"})
        return JSONResponse(
            {
                "status": "ok",
                "version": __version__,
                "cache": cache.backend,
                "providers": sorted(registry.providers),
            }
        )

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(load_html())

    @app.get("/api/stats")
    def stats() -> dict[str, object]:
        summary = store.summary()
        recent = [
            {
                "ts": event.ts,
                "model": event.model_id,
                "provider": event.provider,
                "complexity": event.complexity,
                "policy": event.policy,
                "cached": event.cached,
                "escalated": event.escalated,
                "failover_count": event.failover_count,
                "latency_ms": round(event.latency_ms, 2),
                "cost_usd": event.cost_usd,
                "cost_saved_usd": event.cost_saved_usd,
            }
            for event in store.recent_usage(20)
        ]
        summary["recent"] = recent
        summary["cache_backend"] = cache.backend
        summary["providers"] = sorted(registry.providers)
        return summary

    @app.get("/metrics")
    def prometheus_metrics() -> Response:
        return Response(content=metrics.render(), media_type=metrics_content_type())

    @app.get("/v1/models")
    def models(_key: Annotated[KeyRecord, Depends(require_key)]) -> dict[str, object]:
        data: list[dict[str, str]] = [{"id": "auto", "object": "model", "owned_by": "gateway"}]
        for model in registry.models:
            data.append({"id": model.id, "object": "model", "owned_by": model.provider})
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    async def chat(
        body: ChatRequest,
        key: Annotated[KeyRecord, Depends(require_key)],
        x_routing_policy: Annotated[str | None, Header()] = None,
    ) -> Response:
        completion = await service.complete(body, key, policy_header=x_routing_policy)
        headers = response_headers(completion)
        if body.stream:
            return StreamingResponse(
                iter_sse(completion),
                media_type="text/event-stream",
                headers=headers,
            )
        return JSONResponse(completion_payload(completion), headers=headers)

    @app.post("/admin/keys", status_code=201)
    def create_key(
        body: KeyCreate,
        _admin: Annotated[KeyRecord, Depends(require_admin)],
    ) -> dict[str, object]:
        raw = "gw_" + secrets.token_urlsafe(24)
        record = store.insert_key(
            raw,
            name=body.name,
            quota_tokens=body.quota_tokens,
            is_admin=False,
        )
        return {
            "api_key": raw,
            "name": record.name,
            "prefix": record.key_prefix,
            "quota_tokens": record.quota_tokens,
        }

    @app.get("/admin/keys")
    def list_keys(_admin: Annotated[KeyRecord, Depends(require_admin)]) -> dict[str, object]:
        return {
            "data": [
                {
                    "id": record.id,
                    "name": record.name,
                    "prefix": record.key_prefix,
                    "quota_tokens": record.quota_tokens,
                    "used_tokens": record.used_tokens,
                    "is_admin": record.is_admin,
                    "disabled": record.disabled,
                }
                for record in store.list_keys()
            ]
        }

    @app.get("/admin/usage")
    def admin_usage(_admin: Annotated[KeyRecord, Depends(require_admin)]) -> dict[str, object]:
        return {
            "summary": store.summary(),
            "recent": [
                {
                    "id": event.id,
                    "key_id": event.key_id,
                    "ts": event.ts,
                    "model": event.model_id,
                    "provider": event.provider,
                    "complexity": event.complexity,
                    "policy": event.policy,
                    "cached": event.cached,
                    "escalated": event.escalated,
                    "cost_usd": event.cost_usd,
                    "cost_saved_usd": event.cost_saved_usd,
                    "latency_ms": event.latency_ms,
                }
                for event in store.recent_usage(50)
            ],
        }

    return app


def require_key(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> KeyRecord:
    if authorization is None or not authorization.lower().startswith("bearer "):
        raise GatewayError(
            "Missing API key. Send Authorization: Bearer <key>.",
            401,
            "authentication_error",
            "missing_api_key",
        )
    raw = authorization.split(" ", 1)[1].strip()
    if not raw:
        raise GatewayError("Missing API key", 401, "authentication_error", "missing_api_key")
    record = request.app.state.store.get_by_hash(hash_key(raw))
    if record is None or record.disabled:
        raise GatewayError("Invalid API key", 401, "authentication_error", "invalid_api_key")
    return record


def require_admin(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> KeyRecord:
    record = require_key(request, authorization)
    if not record.is_admin:
        raise GatewayError("Admin key required", 403, "permission_error", "admin_required")
    return record
