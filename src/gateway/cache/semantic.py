"""Exact match plus cosine-similarity cache.

Entries are scoped to a tenant (API key). A global cache would raise the hit
rate and leak one customer's completion to another.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from gateway.cache.embeddings import Embedder, cosine, tokenize

logger = logging.getLogger(__name__)

_REDIS_KEY = "gw:semantic-cache"


@dataclass
class CacheHit:
    payload: dict[str, Any]
    similarity: float
    kind: str


@dataclass
class _Entry:
    exact_key: str
    tenant: str
    policy: str
    temperature: float
    max_tokens: int
    embedding: list[float]
    text: str
    payload: dict[str, Any]
    expires_at: float


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        *,
        ttl_seconds: int,
        threshold: float,
        max_entries: int = 500,
        min_token_overlap: float = 0.5,
        redis_url: str | None = None,
        redis_client: Any | None = None,
        connect_retries: int = 2,
    ) -> None:
        self.embedder = embedder
        self.ttl_seconds = ttl_seconds
        self.threshold = threshold
        self.max_entries = max_entries
        self.min_token_overlap = min_token_overlap
        self.backend = "memory"
        self._entries: list[_Entry] = []
        self._lock = threading.Lock()
        self._redis: Any | None = redis_client
        if self._redis is None and redis_url:
            self._redis = _connect_redis(redis_url, connect_retries)
        if self._redis is not None:
            self.backend = "redis"

    def get(
        self,
        *,
        exact_key: str,
        text: str,
        tenant: str,
        policy: str,
        temperature: float,
        max_tokens: int,
    ) -> CacheHit | None:
        now = time.time()
        vector = self.embedder.embed(text)
        with self._lock:
            entries = [entry for entry in self._load() if entry.expires_at > now]
            exact = self._find_exact(
                entries,
                exact_key=exact_key,
                tenant=tenant,
                policy=policy,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if exact is not None:
                return CacheHit(payload=exact.payload, similarity=1.0, kind="exact")
            best: _Entry | None = None
            best_score = self.threshold
            for entry in entries:
                if not _same_scope(entry, tenant, policy, temperature, max_tokens):
                    continue
                # Overlap rejects hash collisions between unrelated short prompts.
                if token_overlap(text, entry.text) < self.min_token_overlap:
                    continue
                score = cosine(vector, entry.embedding)
                if score >= best_score:
                    best = entry
                    best_score = score
            self._store(entries)
            if best is None:
                return None
            return CacheHit(payload=best.payload, similarity=best_score, kind="semantic")

    def set(
        self,
        *,
        exact_key: str,
        text: str,
        tenant: str,
        policy: str,
        temperature: float,
        max_tokens: int,
        payload: dict[str, Any],
    ) -> None:
        now = time.time()
        entry = _Entry(
            exact_key=exact_key,
            tenant=tenant,
            policy=policy,
            temperature=round(temperature, 3),
            max_tokens=max_tokens,
            embedding=self.embedder.embed(text),
            text=text,
            payload=payload,
            expires_at=now + self.ttl_seconds,
        )
        with self._lock:
            entries = [item for item in self._load() if item.expires_at > now]
            entries = [item for item in entries if item.exact_key != exact_key]
            entries.append(entry)
            if len(entries) > self.max_entries:
                entries = entries[-self.max_entries :]
            self._store(entries)

    def _find_exact(
        self,
        entries: list[_Entry],
        *,
        exact_key: str,
        tenant: str,
        policy: str,
        temperature: float,
        max_tokens: int,
    ) -> _Entry | None:
        for entry in entries:
            if entry.exact_key != exact_key:
                continue
            if _same_scope(entry, tenant, policy, temperature, max_tokens):
                return entry
        return None

    def _load(self) -> list[_Entry]:
        if self._redis is None:
            return list(self._entries)
        try:
            raw = self._redis.get(_REDIS_KEY)
        except Exception:
            logger.warning("redis cache read failed; using memory")
            self._redis = None
            self.backend = "memory"
            return list(self._entries)
        if not raw:
            return []
        if isinstance(raw, bytes):
            raw = raw.decode()
        data = json.loads(raw)
        return [_entry_from_dict(item) for item in data]

    def _store(self, entries: list[_Entry]) -> None:
        if self._redis is None:
            self._entries = entries
            return
        try:
            encoded = json.dumps([_entry_to_dict(entry) for entry in entries])
            self._redis.set(_REDIS_KEY, encoded)
        except Exception:
            logger.warning("redis cache write failed; using memory")
            self._redis = None
            self.backend = "memory"
            self._entries = entries


def _same_scope(
    entry: _Entry,
    tenant: str,
    policy: str,
    temperature: float,
    max_tokens: int,
) -> bool:
    return (
        entry.tenant == tenant
        and entry.policy == policy
        and entry.max_tokens == max_tokens
        and abs(entry.temperature - round(temperature, 3)) < 1e-6
    )


def _entry_to_dict(entry: _Entry) -> dict[str, Any]:
    return {
        "exact_key": entry.exact_key,
        "tenant": entry.tenant,
        "policy": entry.policy,
        "temperature": entry.temperature,
        "max_tokens": entry.max_tokens,
        "embedding": entry.embedding,
        "text": entry.text,
        "payload": entry.payload,
        "expires_at": entry.expires_at,
    }


def _entry_from_dict(data: dict[str, Any]) -> _Entry:
    return _Entry(
        exact_key=str(data["exact_key"]),
        tenant=str(data["tenant"]),
        policy=str(data["policy"]),
        temperature=float(data["temperature"]),
        max_tokens=int(data["max_tokens"]),
        embedding=[float(value) for value in data["embedding"]],
        text=str(data.get("text", "")),
        payload=dict(data["payload"]),
        expires_at=float(data["expires_at"]),
    )


def token_overlap(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    shared = left_tokens & right_tokens
    return len(shared) / len(left_tokens | right_tokens)


def _connect_redis(url: str, retries: int) -> Any | None:
    try:
        import redis
    except ImportError:
        logger.warning("redis package is not installed; using in-memory cache")
        return None
    client = redis.Redis.from_url(url, socket_connect_timeout=0.3, socket_timeout=0.3)
    attempts = max(1, retries)
    for _ in range(attempts):
        try:
            client.ping()
            return client
        except Exception:
            time.sleep(0.05)
    logger.warning("redis unavailable at %s; using in-memory cache", url)
    return None
