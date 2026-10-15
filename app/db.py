"""asyncpg pool. One pool per process, created on startup."""

from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from .config import DATABASE_URL

_pool: asyncpg.Pool | None = None


async def _init(conn: asyncpg.Connection) -> None:
    # Teaches asyncpg the pgvector type, so a numpy array binds to vector(384)
    # directly instead of being stringified.
    await register_vector(conn)


async def connect() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL, min_size=2, max_size=10, init=_init,
            command_timeout=10,
        )
    return _pool


async def close() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("db.connect() has not run -- check the lifespan handler")
    return _pool
