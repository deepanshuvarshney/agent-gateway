"""Classifier, policies, and the verifier."""

from gateway.cache.embeddings import HashEmbedder
from gateway.catalog import ModelSpec, label_tier, tier_label
from gateway.routing.classifier import Classification, classify_text, maybe_tiebreak
from gateway.routing.features import prompt_tier
from gateway.routing.policies import choose_model
from gateway.routing.verifier import verify_answer


def _model(
    model_id: str,
    *,
    tier: int,
    cost: float,
    latency: int,
    quality: int,
    provider: str = "mock",
) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        provider=provider,
        upstream_id=model_id,
        tier=tier,
        input_cost_per_million=cost,
        output_cost_per_million=cost,
        latency_rank=latency,
        quality_rank=quality,
    )


def test_classifier_labels_the_benchmark_shapes() -> None:
    easy = classify_text("What is the capital of France?")
    medium = classify_text("Explain how DNS resolution works in a short paragraph with an example.")
    hard = classify_text("Implement Dijkstra's algorithm and analyze its complexity.")
    assert (easy.label, medium.label, hard.label) == ("easy", "medium", "hard")
    assert easy.source == "heuristic"
    assert (
        prompt_tier(hard.label and "Implement Dijkstra's algorithm and analyze its complexity.")
        == 2
    )


def test_embedding_tiebreak_moves_one_tier() -> None:
    class Fixed:
        def embed(self, text: str) -> list[float]:
            if "hard prototype" in text or "dijkstra" in text:
                return [1.0, 0.0, 0.0]
            if text in {
                "implement a thread-safe lru cache and analyze its complexity",
                "prove that the square root of 2 is irrational step by step",
                "compare and contrast the trade-off of two algorithms and implement one",
            }:
                return [1.0, 0.0, 0.0]
            return [0.0, 1.0, 0.0]

    label, source = maybe_tiebreak("medium", 2.7, "dijkstra proof", Fixed())
    assert (label, source) == ("hard", "embedding-tiebreak")
    unchanged, source = maybe_tiebreak("easy", -2.0, "dijkstra proof", Fixed())
    assert unchanged == "easy"
    assert source == "heuristic"
    same, source = maybe_tiebreak("medium", 1.2, "hello", None)
    assert same == "medium"


def test_classify_with_hash_embedder_stays_on_clear_prompts() -> None:
    result = classify_text("What is a noun?", HashEmbedder())
    assert result == Classification(result.label, result.score, result.source)
    assert result.label == "easy"


def test_policies_prefer_cost_latency_or_quality() -> None:
    cheap_slow = _model("cheap", tier=0, cost=0.1, latency=5, quality=1)
    fast_pricey = _model("fast", tier=0, cost=5, latency=1, quality=2)
    strong = _model("strong", tier=2, cost=9, latency=3, quality=9)
    models = [fast_pricey, cheap_slow, strong]
    assert choose_model("cost", models, min_tier=0).id == "cheap"  # type: ignore[union-attr]
    assert choose_model("latency", models, min_tier=0).id == "fast"  # type: ignore[union-attr]
    assert choose_model("quality", models, min_tier=0).id == "strong"  # type: ignore[union-attr]
    assert choose_model("explicit", models, min_tier=0, requested="fast").id == "fast"  # type: ignore[union-attr]
    assert choose_model("explicit", models, min_tier=0, requested="missing") is None
    assert choose_model("cost", [], min_tier=0) is None
    # No model meets tier 3, so cost falls back to the strongest available.
    assert choose_model("cost", models, min_tier=3).id == "strong"  # type: ignore[union-attr]
    assert choose_model("latency", [cheap_slow], min_tier=5).id == "cheap"  # type: ignore[union-attr]


def test_unknown_policy_and_tier_helpers() -> None:
    try:
        choose_model("nope", [_model("a", tier=0, cost=1, latency=1, quality=1)], min_tier=0)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    assert tier_label(0) == "easy"
    assert tier_label(1) == "medium"
    assert tier_label(4) == "hard"
    assert label_tier("hard") == 2
    assert label_tier("nope") == 1


def test_verifier_thresholds() -> None:
    assert verify_answer("What is a noun?", "Short reply abc.")[0] is True
    assert verify_answer("What is a noun?", "tiny")[0] is False
    medium = "Explain how DNS resolution works in a short paragraph with an example."
    assert verify_answer(medium, "too short")[0] is False
    long_medium = "x" * 90
    assert verify_answer(medium, long_medium)[0] is True
    hard = "Prove this theorem step by step."
    assert verify_answer(hard, "x" * 200) == (False, "insufficient_reasoning")
    reasoned = ("because the trade-off is real. " * 8).strip()
    assert verify_answer(hard, reasoned)[0] is True
    assert verify_answer(hard, "")[0] is False
    assert verify_answer("Hello", "[[LOW_QUALITY]] nope")[0] is False
