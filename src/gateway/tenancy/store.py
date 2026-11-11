"""SQLite by default, Postgres when DATABASE_URL points at it."""

import hashlib
import importlib.util
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

metadata = MetaData()

api_keys = Table(
    "api_keys",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String, nullable=False),
    Column("key_hash", String, nullable=False, unique=True),
    Column("key_prefix", String, nullable=False),
    Column("quota_tokens", Integer, nullable=False),
    Column("used_tokens", Integer, nullable=False, default=0),
    Column("is_admin", Boolean, nullable=False, default=False),
    Column("disabled", Boolean, nullable=False, default=False),
    Column("created_at", String, nullable=False),
)

usage_events = Table(
    "usage_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("key_id", Integer, nullable=False),
    Column("ts", String, nullable=False),
    Column("model_id", String, nullable=False),
    Column("provider", String, nullable=False),
    Column("complexity", String, nullable=False),
    Column("policy", String, nullable=False),
    Column("prompt_tokens", Integer, nullable=False),
    Column("completion_tokens", Integer, nullable=False),
    Column("cached", Integer, nullable=False),
    Column("escalated", Integer, nullable=False),
    Column("failover_count", Integer, nullable=False),
    Column("cost_usd", Float, nullable=False),
    Column("cost_saved_usd", Float, nullable=False),
    Column("latency_ms", Float, nullable=False),
)


@dataclass(frozen=True)
class KeyRecord:
    id: int
    name: str
    key_hash: str
    key_prefix: str
    quota_tokens: int
    used_tokens: int
    is_admin: bool
    disabled: bool


@dataclass(frozen=True)
class UsageRecord:
    id: int
    key_id: int
    ts: str
    model_id: str
    provider: str
    complexity: str
    policy: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    escalated: bool
    failover_count: int
    cost_usd: float
    cost_saved_usd: float
    latency_ms: float


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def key_prefix(raw: str) -> str:
    return raw[:12]


def normalize_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url.removeprefix("postgres://")
    if url.startswith("postgresql://") and not url.startswith("postgresql+"):
        return "postgresql+psycopg://" + url.removeprefix("postgresql://")
    return url


def create_db_engine(url: str) -> Engine:
    normalized = normalize_db_url(url)
    if normalized.startswith("postgresql+psycopg") and importlib.util.find_spec("psycopg") is None:
        raise RuntimeError(
            "Postgres support requires the optional extra: pip install 'agent-gateway[postgres]'"
        )
    if normalized.startswith("sqlite"):
        _prepare_sqlite(normalized)
        if ":memory:" in normalized:
            return create_engine(
                normalized,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool,
            )
        return create_engine(normalized, connect_args={"check_same_thread": False})
    return create_engine(normalized, pool_pre_ping=True)


def _prepare_sqlite(url: str) -> None:
    if ":memory:" in url:
        return
    raw = url.removeprefix("sqlite:///")
    if not raw or raw == ":memory:":
        return
    path = Path(raw)
    if str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)


class Store:
    def __init__(self, url: str) -> None:
        self.engine = create_db_engine(url)
        metadata.create_all(self.engine)

    def ping(self) -> None:
        with self.engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")

    def ensure_key(self, raw: str, *, name: str, quota_tokens: int, is_admin: bool) -> KeyRecord:
        existing = self.get_by_hash(hash_key(raw))
        if existing is not None:
            return existing
        return self.insert_key(raw, name=name, quota_tokens=quota_tokens, is_admin=is_admin)

    def insert_key(
        self,
        raw: str,
        *,
        name: str,
        quota_tokens: int,
        is_admin: bool,
    ) -> KeyRecord:
        now = _now()
        with self.engine.begin() as conn:
            result = conn.execute(
                insert(api_keys)
                .values(
                    name=name,
                    key_hash=hash_key(raw),
                    key_prefix=key_prefix(raw),
                    quota_tokens=quota_tokens,
                    used_tokens=0,
                    is_admin=is_admin,
                    disabled=False,
                    created_at=now,
                )
                .returning(api_keys.c.id)
            )
            new_id = int(result.scalar_one())
        created = self.get_by_hash(hash_key(raw))
        if created is None:
            raise RuntimeError(f"failed to read key {new_id} after insert")
        return created

    def get_by_hash(self, digest: str) -> KeyRecord | None:
        with self.engine.connect() as conn:
            row = conn.execute(select(api_keys).where(api_keys.c.key_hash == digest)).first()
        if row is None:
            return None
        return _key(row)

    def set_disabled(self, key_id: int, disabled: bool) -> None:
        with self.engine.begin() as conn:
            conn.execute(update(api_keys).where(api_keys.c.id == key_id).values(disabled=disabled))

    def list_keys(self) -> list[KeyRecord]:
        with self.engine.connect() as conn:
            rows = conn.execute(select(api_keys).order_by(api_keys.c.id)).all()
        return [_key(row) for row in rows]

    def add_usage(
        self,
        key_id: int,
        *,
        model_id: str,
        provider: str,
        complexity: str,
        policy: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached: bool,
        escalated: bool,
        failover_count: int,
        cost_usd: float,
        cost_saved_usd: float,
        latency_ms: float,
        charge_tokens: int,
    ) -> None:
        with self.engine.begin() as conn:
            conn.execute(
                update(api_keys)
                .where(api_keys.c.id == key_id)
                .values(used_tokens=api_keys.c.used_tokens + charge_tokens)
            )
            conn.execute(
                insert(usage_events).values(
                    key_id=key_id,
                    ts=_now(),
                    model_id=model_id,
                    provider=provider,
                    complexity=complexity,
                    policy=policy,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cached=1 if cached else 0,
                    escalated=1 if escalated else 0,
                    failover_count=failover_count,
                    cost_usd=cost_usd,
                    cost_saved_usd=cost_saved_usd,
                    latency_ms=latency_ms,
                )
            )

    def recent_usage(self, limit: int = 20) -> list[UsageRecord]:
        with self.engine.connect() as conn:
            rows = conn.execute(
                select(usage_events).order_by(usage_events.c.id.desc()).limit(limit)
            ).all()
        return [_usage(row) for row in rows]

    def summary(self) -> dict[str, object]:
        with self.engine.connect() as conn:
            totals = conn.execute(
                select(
                    usage_events.c.cached,
                    usage_events.c.cost_usd,
                    usage_events.c.cost_saved_usd,
                    usage_events.c.escalated,
                    usage_events.c.failover_count,
                    usage_events.c.complexity,
                    usage_events.c.model_id,
                )
            ).all()
        requests = len(totals)
        cache_hits = sum(int(row.cached) for row in totals)
        cost = float(sum(row.cost_usd for row in totals))
        saved = float(sum(row.cost_saved_usd for row in totals))
        escalations = sum(int(row.escalated) for row in totals)
        failovers = int(sum(row.failover_count for row in totals))
        by_complexity: dict[str, int] = {}
        by_model: dict[str, int] = {}
        for row in totals:
            by_complexity[row.complexity] = by_complexity.get(row.complexity, 0) + 1
            by_model[row.model_id] = by_model.get(row.model_id, 0) + 1
        hit_rate = (cache_hits / requests) if requests else 0.0
        baseline = cost + saved
        savings_pct = (saved / baseline * 100) if baseline > 0 else 0.0
        return {
            "requests": requests,
            "cache_hits": cache_hits,
            "cache_hit_rate": hit_rate,
            "cost_usd": cost,
            "cost_saved_usd": saved,
            "savings_pct": savings_pct,
            "escalations": escalations,
            "failovers": failovers,
            "by_complexity": by_complexity,
            "by_model": by_model,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _key(row: object) -> KeyRecord:
    return KeyRecord(
        id=int(row.id),  # type: ignore[attr-defined]
        name=str(row.name),  # type: ignore[attr-defined]
        key_hash=str(row.key_hash),  # type: ignore[attr-defined]
        key_prefix=str(row.key_prefix),  # type: ignore[attr-defined]
        quota_tokens=int(row.quota_tokens),  # type: ignore[attr-defined]
        used_tokens=int(row.used_tokens),  # type: ignore[attr-defined]
        is_admin=bool(row.is_admin),  # type: ignore[attr-defined]
        disabled=bool(row.disabled),  # type: ignore[attr-defined]
    )


def _usage(row: object) -> UsageRecord:
    return UsageRecord(
        id=int(row.id),  # type: ignore[attr-defined]
        key_id=int(row.key_id),  # type: ignore[attr-defined]
        ts=str(row.ts),  # type: ignore[attr-defined]
        model_id=str(row.model_id),  # type: ignore[attr-defined]
        provider=str(row.provider),  # type: ignore[attr-defined]
        complexity=str(row.complexity),  # type: ignore[attr-defined]
        policy=str(row.policy),  # type: ignore[attr-defined]
        prompt_tokens=int(row.prompt_tokens),  # type: ignore[attr-defined]
        completion_tokens=int(row.completion_tokens),  # type: ignore[attr-defined]
        cached=bool(row.cached),  # type: ignore[attr-defined]
        escalated=bool(row.escalated),  # type: ignore[attr-defined]
        failover_count=int(row.failover_count),  # type: ignore[attr-defined]
        cost_usd=float(row.cost_usd),  # type: ignore[attr-defined]
        cost_saved_usd=float(row.cost_saved_usd),  # type: ignore[attr-defined]
        latency_ms=float(row.latency_ms),  # type: ignore[attr-defined]
    )
