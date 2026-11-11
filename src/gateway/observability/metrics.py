"""Prometheus metrics for latency, tokens, cache, cost, and failover."""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)


class GatewayMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.requests = Counter(
            "gateway_requests_total",
            "Completed chat requests.",
            ["policy", "complexity", "cached", "provider"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "gateway_request_latency_seconds",
            "End-to-end gateway latency.",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
            registry=self.registry,
        )
        self.tokens = Counter(
            "gateway_tokens_total",
            "Prompt and completion tokens.",
            ["direction"],
            registry=self.registry,
        )
        self.cache_events = Counter(
            "gateway_cache_events_total",
            "Cache lookups by result.",
            ["result"],
            registry=self.registry,
        )
        self.cost = Counter(
            "gateway_cost_usd_total",
            "Estimated spend using planning prices.",
            registry=self.registry,
        )
        self.cost_saved = Counter(
            "gateway_cost_saved_usd_total",
            "Estimated savings versus always using the strongest model.",
            registry=self.registry,
        )
        self.failovers = Counter(
            "gateway_failovers_total",
            "Provider switches after rate limit, circuit open, or 429/5xx.",
            registry=self.registry,
        )
        self.escalations = Counter(
            "gateway_escalations_total",
            "Verifier-triggered reroutes to a stronger model.",
            registry=self.registry,
        )
        self.errors = Counter(
            "gateway_errors_total",
            "Rejected or failed requests.",
            ["code"],
            registry=self.registry,
        )

    def observe_success(
        self,
        *,
        policy: str,
        complexity: str,
        cached: bool,
        cache_kind: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        cost_saved_usd: float,
        failover_count: int,
        escalated: bool,
        latency_seconds: float,
    ) -> None:
        self.requests.labels(
            policy=policy,
            complexity=complexity,
            cached="true" if cached else "false",
            provider=provider,
        ).inc()
        self.latency.observe(latency_seconds)
        self.tokens.labels(direction="prompt").inc(prompt_tokens)
        self.tokens.labels(direction="completion").inc(completion_tokens)
        self.cache_events.labels(result=cache_kind).inc()
        self.cost.inc(cost_usd)
        self.cost_saved.inc(cost_saved_usd)
        if failover_count:
            self.failovers.inc(failover_count)
        if escalated:
            self.escalations.inc()

    def observe_error(self, code: str, latency_seconds: float) -> None:
        self.errors.labels(code=code).inc()
        self.latency.observe(latency_seconds)

    def render(self) -> bytes:
        return generate_latest(self.registry)


def metrics_content_type() -> str:
    return str(CONTENT_TYPE_LATEST)
