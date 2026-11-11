"""Small text helpers shared by routing, caching, and token accounting."""

import re


def message_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(message.get("content", "") for message in messages)


def user_text(messages: list[dict[str, str]]) -> str:
    """Text the classifier should see. System prompts are instructions, not the task."""
    parts = [message.get("content", "") for message in messages if message.get("role") != "system"]
    if parts:
        return "\n".join(parts)
    return message_text(messages)


def canonical_messages(messages: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for message in messages:
        content = re.sub(r"\s+", " ", message.get("content", "")).strip().lower()
        lines.append(f"{message.get('role', '')}:{content}")
    return "\n".join(lines)


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token). Deterministic and offline."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
