"""Routing policies over the enabled model catalog."""

from gateway.catalog import ModelSpec


def choose_model(
    policy: str,
    models: list[ModelSpec],
    *,
    min_tier: int,
    requested: str | None = None,
) -> ModelSpec | None:
    """Pick one model.

    cost: cheapest model whose tier can cover ``min_tier``.
    latency: lowest latency rank among models that cover ``min_tier``.
    quality: highest quality rank, still respecting ``min_tier`` when escalating.
    explicit: the requested id, ignoring tier.
    """
    if not models:
        return None
    if policy == "explicit":
        if not requested:
            return None
        for model in models:
            if model.id == requested or model.upstream_id == requested:
                return model
        return None

    capable = [model for model in models if model.tier >= min_tier]
    pool = capable or sorted(models, key=lambda model: -model.tier)
    if policy == "cost":
        pool = sorted(
            pool,
            key=lambda model: (
                model.input_cost_per_million + model.output_cost_per_million,
                model.latency_rank,
                model.tier,
            ),
        )
        if not capable:
            pool = sorted(
                models,
                key=lambda model: (
                    -model.tier,
                    model.input_cost_per_million + model.output_cost_per_million,
                ),
            )
        return pool[0]
    if policy == "latency":
        pool = sorted(pool, key=lambda model: (model.latency_rank, model.output_cost_per_million))
        if not capable:
            pool = sorted(models, key=lambda model: (-model.tier, model.latency_rank))
        return pool[0]
    if policy == "quality":
        target = capable or models
        target = sorted(target, key=lambda model: (-model.quality_rank, model.latency_rank))
        return target[0]
    raise ValueError(f"Unknown routing policy: {policy}")
