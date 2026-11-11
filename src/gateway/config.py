"""Runtime configuration. Every field can be set with a GATEWAY_ environment variable."""

import os
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Gateway settings.

    Real provider keys are optional. With none set, only the deterministic mock
    provider is used, which is what CI and the free Render deploy rely on.
    """

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "local"
    host: str = "0.0.0.0"
    port: int = 8000
    database_url: str = "sqlite:///./data/gateway.db"
    redis_url: str | None = None
    redis_connect_retries: int = 8
    embeddings: str = "hash"
    embedding_dim: int = 256
    cache_ttl_seconds: int = 3600
    cache_similarity_threshold: float = 0.80
    cache_max_entries: int = 500
    default_policy: str = "cost"
    admin_api_key: str = "gw_admin_dev"
    bootstrap_api_key: str = "gw_demo_public"
    default_quota_tokens: int = 1_000_000
    default_max_tokens: int = 256
    max_tokens_per_request: int = 2048
    max_prompt_chars: int = 20_000
    otel_exporter: str = "none"
    otel_endpoint: str = "http://localhost:4318/v1/traces"
    groq_api_key: str | None = None
    gemini_api_key: str | None = None
    ollama_base_url: str | None = None
    request_timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.05
    circuit_failure_threshold: int = 3
    circuit_reset_seconds: float = 30.0
    provider_rps: float = 20.0
    provider_burst: float = 40.0
    log_level: str = "INFO"
    mock_latency_ms: int = 0
    enable_escalation: bool = True
    injection_mode: str = "block"
    # None means "on only when no real provider is configured".
    enable_mock: bool | None = None

    def resolved_groq_api_key(self) -> str | None:
        return _first(self.groq_api_key, os.environ.get("GROQ_API_KEY"))

    def resolved_gemini_api_key(self) -> str | None:
        return _first(
            self.gemini_api_key,
            os.environ.get("GEMINI_API_KEY"),
            os.environ.get("GOOGLE_API_KEY"),
        )

    def mock_enabled(self) -> bool:
        if self.enable_mock is not None:
            return self.enable_mock
        return not any(
            [
                self.resolved_groq_api_key(),
                self.resolved_gemini_api_key(),
                _blank(self.ollama_base_url),
            ]
        )


def _blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _first(*values: str | None) -> str | None:
    for value in values:
        cleaned = _blank(value)
        if cleaned:
            return cleaned
    return None


@lru_cache
def get_settings() -> Settings:
    return Settings()
