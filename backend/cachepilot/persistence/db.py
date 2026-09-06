from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool

from cachepilot.persistence.schema import Base


def make_engine(database_url: str) -> AsyncEngine:
    if _is_sqlite_memory(database_url):
        # Every new connection to an in-memory SQLite db is a fresh empty db;
        # a single shared connection keeps the schema alive across sessions.
        return create_async_engine(
            database_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
        )
    return create_async_engine(database_url)


async def create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _is_sqlite_memory(url: str) -> bool:
    return url.startswith("sqlite") and (url.endswith("://") or ":memory:" in url)
