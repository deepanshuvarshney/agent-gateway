"""Model catalog.

Prices are planning rates in USD per 1M tokens. They rank models and estimate
savings versus always calling the strongest model. Free-tier invoices are
often $0; the rates are still useful as a stable preference signal.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    id: str
    provider: str
    upstream_id: str
    tier: int
    input_cost_per_million: float
    output_cost_per_million: float
    latency_rank: int
    quality_rank: int

    def cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        raw = (
            prompt_tokens * self.input_cost_per_million
            + completion_tokens * self.output_cost_per_million
        )
        return raw / 1_000_000


def _spec(
    model_id: str,
    provider: str,
    tier: int,
    input_cost: float,
    output_cost: float,
    latency_rank: int,
    quality_rank: int,
    upstream_id: str | None = None,
) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        provider=provider,
        upstream_id=upstream_id or model_id,
        tier=tier,
        input_cost_per_million=input_cost,
        output_cost_per_million=output_cost,
        latency_rank=latency_rank,
        quality_rank=quality_rank,
    )


# Groq ids are the ones that succeed on the owner's account. Llama ids 404 there.
GROQ_MODELS: tuple[ModelSpec, ...] = (
    _spec("openai/gpt-oss-20b", "groq", 0, 0.10, 0.40, 1, 2),
    _spec("qwen/qwen3.8-27b", "groq", 1, 0.25, 0.70, 2, 3),
    _spec("openai/gpt-oss-120b", "groq", 2, 0.80, 2.00, 3, 5),
)

GEMINI_MODELS: tuple[ModelSpec, ...] = (
    _spec("gemini-2.0-flash-lite", "gemini", 0, 0.05, 0.20, 2, 2),
    _spec("gemini-2.0-flash", "gemini", 1, 0.15, 0.60, 2, 3),
    _spec("gemini-2.5-flash", "gemini", 2, 0.30, 2.50, 3, 5),
)

OLLAMA_MODELS: tuple[ModelSpec, ...] = (
    _spec("ollama/llama3.2:1b", "ollama", 0, 0.0, 0.0, 4, 1, "llama3.2:1b"),
    _spec("ollama/llama3.2:3b", "ollama", 1, 0.0, 0.0, 5, 2, "llama3.2:3b"),
    _spec("ollama/qwen2.5:7b", "ollama", 2, 0.0, 0.0, 6, 3, "qwen2.5:7b"),
)

MOCK_MODELS: tuple[ModelSpec, ...] = (
    _spec("mock-small", "mock", 0, 0.10, 0.40, 1, 1),
    _spec("mock-medium", "mock", 1, 0.30, 0.80, 2, 2),
    _spec("mock-large", "mock", 2, 1.00, 3.00, 3, 3),
)

TIER_LABELS = ("easy", "medium", "hard")


def tier_label(tier: int) -> str:
    if tier <= 0:
        return "easy"
    if tier == 1:
        return "medium"
    return "hard"


def label_tier(label: str) -> int:
    try:
        return TIER_LABELS.index(label)
    except ValueError:
        return 1
