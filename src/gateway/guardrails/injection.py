"""Cheap prompt-injection checks.

These patterns catch the obvious instruction-override attacks. They will also
flag a prompt that is *about* those phrases. `injection_mode=annotate` keeps
the request and marks the content as untrusted instead of rejecting it.
"""

import re

_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"ignore (all |any )?(previous|prior|above) (instructions|prompts|rules)",
        re.IGNORECASE,
    ),
    re.compile(
        r"disregard (the )?(system|developer|previous) (prompt|message|instructions)",
        re.IGNORECASE,
    ),
    re.compile(
        r"forget (all )?(previous|prior) (instructions|rules)",
        re.IGNORECASE,
    ),
    re.compile(
        r"override (the )?(system|developer) (prompt|instructions)",
        re.IGNORECASE,
    ),
    re.compile(r"reveal (your )?(system|hidden|initial) prompt", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"do anything now", re.IGNORECASE),
    re.compile(r"<\s*/?\s*system\s*>", re.IGNORECASE),
)


def injection_hits(text: str) -> list[str]:
    return [pattern.pattern for pattern in _PATTERNS if pattern.search(text)]
