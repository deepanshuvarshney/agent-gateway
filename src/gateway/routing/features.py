"""Surface features used by the router and, independently, by the verifier."""

import re
from dataclasses import dataclass

HARD_PHRASES: tuple[str, ...] = (
    "prove",
    "proof",
    "derive",
    "step by step",
    "analyze",
    "analyse",
    "trade-off",
    "tradeoff",
    "trade off",
    "optimize",
    "optimise",
    "implement",
    "algorithm",
    "debug",
    "refactor",
    "complexity",
    "big-o",
    "big o",
    "architect",
    "formal",
    "theorem",
    "compare and contrast",
    "failure mode",
)

MEDIUM_PHRASES: tuple[str, ...] = (
    "explain",
    "describe",
    "summarize",
    "summarise",
    "overview",
    "outline",
    "difference between",
    "causes of",
    "purpose of",
    "in a paragraph",
    "for a student",
)

CODE_MARKERS: tuple[str, ...] = (
    "```",
    "def ",
    "class ",
    "function ",
    "import ",
    "public static",
    "fn ",
    "select ",
)


@dataclass(frozen=True)
class Features:
    word_count: int
    has_hard: bool
    has_medium: bool
    has_code: bool
    question_marks: int


def extract(text: str) -> Features:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9_]+", lowered)
    return Features(
        word_count=len(words),
        has_hard=any(phrase in lowered for phrase in HARD_PHRASES),
        has_medium=any(phrase in lowered for phrase in MEDIUM_PHRASES),
        has_code=("```" in text) or any(marker in lowered for marker in CODE_MARKERS),
        question_marks=text.count("?"),
    )


def prompt_tier(text: str) -> int:
    """Independent difficulty signal. The verifier uses this, not the router's label."""
    features = extract(text)
    if features.has_hard or features.has_code:
        return 2
    if features.has_medium:
        return 1
    return 0
