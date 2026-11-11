"""Complexity classifier.

Heuristics decide clear cases. Embedding similarity to class prototypes is only
consulted when the heuristic score sits on a boundary, so a paraphrase can
break a tie without overriding an obvious easy or hard prompt.
"""

from dataclasses import dataclass

from gateway.cache.embeddings import Embedder, cosine
from gateway.routing.features import Features, extract

EASY_PROTOTYPES: tuple[str, ...] = (
    "what is the capital of france",
    "define photosynthesis",
    "hello how are you",
    "who wrote hamlet",
)

MEDIUM_PROTOTYPES: tuple[str, ...] = (
    "explain how dns resolution works with an example",
    "summarize the water cycle for a student",
    "describe the difference between a list and a tuple",
)

HARD_PROTOTYPES: tuple[str, ...] = (
    "implement a thread-safe lru cache and analyze its complexity",
    "prove that the square root of 2 is irrational step by step",
    "compare and contrast the trade-off of two algorithms and implement one",
)

_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "easy": EASY_PROTOTYPES,
    "medium": MEDIUM_PROTOTYPES,
    "hard": HARD_PROTOTYPES,
}

_BOUNDARY_EASY = 1.0
_BOUNDARY_HARD = 3.0
_TIE_BAND = 0.45
_TIE_SIMILARITY = 0.42


@dataclass(frozen=True)
class Classification:
    label: str
    score: float
    source: str


def heuristic_score(features: Features) -> float:
    score = 0.0
    if features.has_hard:
        score += 3.6
    elif features.has_medium:
        score += 1.8
    else:
        score -= 0.5
    if features.has_code:
        score += 1.0
    if features.word_count <= 12 and not features.has_hard and not features.has_medium:
        score -= 1.0
    elif features.word_count >= 45:
        score += 0.7
    if features.question_marks >= 2:
        score += 0.4
    return score


def label_from_score(score: float) -> str:
    if score < _BOUNDARY_EASY:
        return "easy"
    if score < _BOUNDARY_HARD:
        return "medium"
    return "hard"


def nearest_prototype(embedder: Embedder, text: str) -> tuple[str, float]:
    vector = embedder.embed(text)
    best_label = "medium"
    best_sim = -1.0
    for label, samples in _PROTOTYPES.items():
        sims = [cosine(vector, embedder.embed(sample)) for sample in samples]
        average = sum(sims) / len(sims)
        if average > best_sim:
            best_sim = average
            best_label = label
    return best_label, best_sim


def maybe_tiebreak(
    label: str,
    score: float,
    text: str,
    embedder: Embedder | None,
) -> tuple[str, str]:
    if embedder is None:
        return label, "heuristic"
    distance = min(abs(score - _BOUNDARY_EASY), abs(score - _BOUNDARY_HARD))
    if distance > _TIE_BAND:
        return label, "heuristic"
    alternative, similarity = nearest_prototype(embedder, text)
    if similarity < _TIE_SIMILARITY or alternative == label:
        return label, "heuristic"
    # Move at most one tier, so a noisy embedding cannot jump easy -> hard.
    order = ("easy", "medium", "hard")
    if abs(order.index(alternative) - order.index(label)) != 1:
        return label, "heuristic"
    return alternative, "embedding-tiebreak"


def classify_text(text: str, embedder: Embedder | None = None) -> Classification:
    features = extract(text)
    score = heuristic_score(features)
    label = label_from_score(score)
    label, source = maybe_tiebreak(label, score, text, embedder)
    return Classification(label=label, score=score, source=source)
