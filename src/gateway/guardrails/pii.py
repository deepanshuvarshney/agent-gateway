"""Redact common PII before a prompt is logged, cached, or sent upstream."""

import re

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_PHONE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b")
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")


def luhn_ok(number: str) -> bool:
    digits = [int(char) for char in number if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def redact_text(text: str) -> str:
    def replace_card(match: re.Match[str]) -> str:
        if luhn_ok(match.group(0)):
            return "[REDACTED_CARD]"
        return match.group(0)

    redacted = _CARD.sub(replace_card, text)
    redacted = _SSN.sub("[REDACTED_SSN]", redacted)
    redacted = _PHONE.sub("[REDACTED_PHONE]", redacted)
    redacted = _EMAIL.sub("[REDACTED_EMAIL]", redacted)
    return redacted


def redact_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {"role": message.get("role", "user"), "content": redact_text(message.get("content", ""))}
        for message in messages
    ]
