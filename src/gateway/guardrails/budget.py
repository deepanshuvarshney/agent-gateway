"""Per-request and per-key token budgets."""

from gateway.config import Settings
from gateway.errors import GatewayError
from gateway.tenancy.store import KeyRecord


def assert_budget(
    *,
    max_tokens: int,
    prompt_tokens: int,
    key: KeyRecord,
    settings: Settings,
) -> None:
    if max_tokens < 1:
        raise GatewayError(
            "max_tokens must be at least 1",
            400,
            "invalid_request_error",
            "max_tokens",
        )
    if max_tokens > settings.max_tokens_per_request:
        raise GatewayError(
            f"max_tokens exceeds the per-request cap of {settings.max_tokens_per_request}",
            400,
            "invalid_request_error",
            "max_tokens",
        )
    projected = key.used_tokens + prompt_tokens + max_tokens
    if key.used_tokens >= key.quota_tokens or projected > key.quota_tokens:
        raise GatewayError(
            "API key token quota exceeded",
            429,
            "rate_limit_error",
            "quota_exceeded",
        )
