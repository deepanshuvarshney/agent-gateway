"""Shared fixtures. Tests never call a real model provider."""

import os

# Isolate the process before gateway settings are cached.
for _name in (
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "GATEWAY_GROQ_API_KEY",
    "GATEWAY_GEMINI_API_KEY",
    "GATEWAY_OLLAMA_BASE_URL",
    "GATEWAY_REDIS_URL",
):
    os.environ.pop(_name, None)
os.environ["GATEWAY_OTEL_EXPORTER"] = "none"
os.environ["GATEWAY_DATABASE_URL"] = "sqlite:///:memory:"
os.environ["GATEWAY_EMBEDDINGS"] = "hash"
os.environ["GATEWAY_ENABLE_MOCK"] = "true"

import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from gateway.config import Settings

AUTH = {"Authorization": "Bearer test-key"}
ADMIN = {"Authorization": "Bearer admin-test"}


@pytest.fixture
def settings(tmp_path: pytest.TempPathFactory) -> Settings:
    return Settings(
        _env_file=None,
        database_url=f"sqlite:///{tmp_path}/gateway.db",
        redis_url=None,
        otel_exporter="none",
        embeddings="hash",
        admin_api_key="admin-test",
        bootstrap_api_key="test-key",
        enable_mock=True,
        groq_api_key=None,
        gemini_api_key=None,
        ollama_base_url=None,
        retry_backoff_seconds=0,
        mock_latency_ms=0,
        log_level="WARNING",
        default_quota_tokens=1_000_000,
    )


@pytest.fixture
def client(settings: Settings) -> TestClient:
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
