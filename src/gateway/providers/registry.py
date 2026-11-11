"""Which providers and models are active for this process."""

from gateway.catalog import (
    GEMINI_MODELS,
    GROQ_MODELS,
    MOCK_MODELS,
    OLLAMA_MODELS,
    ModelSpec,
)
from gateway.config import Settings
from gateway.providers.base import BaseProvider
from gateway.providers.mock import MockProvider
from gateway.providers.ollama import OllamaProvider
from gateway.providers.openai_compat import OpenAICompatProvider

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class Registry:
    def __init__(self, providers: dict[str, BaseProvider], models: list[ModelSpec]) -> None:
        self.providers = providers
        self.models = models
        self._by_id = {model.id: model for model in models}

    def get(self, model_id: str) -> ModelSpec | None:
        return self._by_id.get(model_id)

    def provider(self, name: str) -> BaseProvider:
        return self.providers[name]

    def has_real_provider(self) -> bool:
        return any(name != "mock" for name in self.providers)

    def routing_models(self, *, explicit: bool) -> list[ModelSpec]:
        """Automatic policies skip the mock when a real provider is configured.

        Explicit `mock-*` requests still resolve, which keeps the offline demo
        available next to a live key.
        """
        if explicit or not self.has_real_provider():
            return list(self.models)
        real = [model for model in self.models if model.provider != "mock"]
        return real or list(self.models)

    def failover_chain(self, model_id: str) -> list[ModelSpec]:
        primary = self.get(model_id)
        if primary is None:
            return []
        others = [
            model
            for model in self.models
            if model.id != primary.id and model.provider in self.providers
        ]
        same = [model for model in others if model.tier == primary.tier]
        higher = [model for model in others if model.tier > primary.tier]
        same.sort(key=lambda model: (model.latency_rank, model.output_cost_per_million))
        higher.sort(key=lambda model: (model.tier, model.latency_rank))
        return [primary, *same, *higher]

    def biggest(self) -> ModelSpec:
        pool = self.routing_models(explicit=False)
        return max(pool, key=lambda model: (model.quality_rank, model.output_cost_per_million))

    async def aclose(self) -> None:
        for provider in self.providers.values():
            await provider.aclose()


def build_registry(settings: Settings) -> Registry:
    providers: dict[str, BaseProvider] = {}
    models: list[ModelSpec] = []
    if settings.mock_enabled():
        providers["mock"] = MockProvider(latency_ms=settings.mock_latency_ms)
        models.extend(MOCK_MODELS)
    groq_key = settings.resolved_groq_api_key()
    if groq_key:
        providers["groq"] = OpenAICompatProvider(
            "groq",
            GROQ_BASE_URL,
            groq_key,
            timeout=settings.request_timeout_seconds,
        )
        models.extend(GROQ_MODELS)
    gemini_key = settings.resolved_gemini_api_key()
    if gemini_key:
        providers["gemini"] = OpenAICompatProvider(
            "gemini",
            GEMINI_BASE_URL,
            gemini_key,
            timeout=settings.request_timeout_seconds,
        )
        models.extend(GEMINI_MODELS)
    if settings.ollama_base_url:
        providers["ollama"] = OllamaProvider(
            settings.ollama_base_url,
            timeout=max(settings.request_timeout_seconds, 60.0),
        )
        models.extend(OLLAMA_MODELS)
    if not providers:
        providers["mock"] = MockProvider(latency_ms=settings.mock_latency_ms)
        models.extend(MOCK_MODELS)
    return Registry(providers, models)
