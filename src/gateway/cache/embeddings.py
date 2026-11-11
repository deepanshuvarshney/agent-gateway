"""Embedding backends.

Hash embeddings are the default: deterministic, offline, and small enough for
the free Render instance. sentence-transformers is optional and imported only
when selected.
"""

import hashlib
import math
import re
from typing import Protocol

_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "of",
    "to",
    "and",
    "in",
    "on",
    "for",
    "with",
    "what",
    "who",
    "why",
    "how",
    "do",
    "does",
    "please",
    "me",
    "you",
    "your",
    "it",
    "this",
    "that",
    "be",
    "as",
    "at",
    "by",
    "or",
    "from",
    "was",
    "were",
    "can",
    "its",
}


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]:
        """Return an L2-normalized vector."""


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if word not in _STOPWORDS and len(word) > 1]


def cosine(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return float(sum(a * b for a, b in zip(left, right, strict=True)))


class HashEmbedder:
    """Signed feature hashing over content tokens. No model download."""

    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("embedding dim must be at least 8")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = tokenize(text)
        if not tokens:
            return vector
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]


class SentenceTransformerEmbedder:
    """Local embedding model. Requires the optional `embeddings` extra."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it with pip install 'agent-gateway[embeddings]'."
            ) from exc
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> list[float]:
        vector = self._model.encode(text, normalize_embeddings=True)
        return [float(value) for value in vector]


def build_embedder(kind: str, dim: int = 256) -> Embedder:
    if kind == "hash":
        return HashEmbedder(dim)
    if kind == "sentence-transformers":
        return SentenceTransformerEmbedder()
    raise ValueError(f"Unknown embeddings backend '{kind}'. Use 'hash' or 'sentence-transformers'.")
