"""Heuristic verifier.

This is intentionally not an LLM-as-judge: it has to run with no API key and
stay deterministic in CI. It does not trust the router's label. It re-reads
the prompt and checks that the draft is deep enough for that prompt.
"""

from gateway.routing.features import prompt_tier


def verify_answer(prompt: str, answer: str) -> tuple[bool, str]:
    stripped = answer.strip()
    if not stripped:
        return False, "empty"
    if "[[LOW_QUALITY]]" in answer:
        return False, "low_quality"
    needed = prompt_tier(prompt)
    if needed >= 2:
        lowered = stripped.lower()
        has_reason = "because" in lowered or "trade-off" in lowered or "tradeoff" in lowered
        if len(stripped) < 160 or not has_reason:
            return False, "insufficient_reasoning"
        return True, "ok"
    if needed == 1:
        if len(stripped) < 80:
            return False, "too_shallow"
        return True, "ok"
    if len(stripped) < 12:
        return False, "too_short"
    return True, "ok"
