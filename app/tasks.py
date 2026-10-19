"""Celery: everything that must NOT be on the request path.

Writing the cache entry is the single most tempting thing to do synchronously
and the single worst. It adds a Postgres insert plus an embedding write to every
miss, on the path the caller is waiting on, purely to benefit some FUTURE
request. Moving it here costs one thing: two identical requests arriving within
a few hundred milliseconds of each other both miss. That is a good trade, and
being able to say so out loud is the point.
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal

import numpy as np
from celery import Celery

from . import accounting, db
from .cache import semantic
from .config import REDIS_URL
from .providers.base import Usage

celery_app = Celery("gateway", broker=REDIS_URL, backend=None)
celery_app.conf.update(
    task_acks_late=True,              # redeliver if a worker dies mid-task
    worker_prefetch_multiplier=1,     # do not hoard tasks into one worker
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    beat_schedule={
        # TTL is enforced at READ time, so expired rows are filtered out but
        # never removed. Without this the table and its index grow forever and
        # the exact-KNN scan degrades -- slowly, invisibly, in production.
        "prune-expired-cache": {"task": "prune_expired_cache", "schedule": 3600.0},
    },
)


def _run(coro):
    """Celery workers are synchronous; our data layer is async. One event loop
    per task keeps them from sharing a pool across threads."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(db.connect())
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(db.close())
        loop.close()


@celery_app.task(name="persist_cache_entry", max_retries=3,
                 autoretry_for=(Exception,), retry_backoff=True)
def persist_cache_entry(tenant_id: int, ns: str, prompt: str, vector: list,
                        response: str, prompt_tokens: int,
                        completion_tokens: int, provider: str, model: str):
    _run(semantic.store(tenant_id, ns, prompt,
                        np.asarray(vector, dtype=np.float32), response,
                        prompt_tokens, completion_tokens, provider, model))


@celery_app.task(name="prune_expired_cache")
def prune_expired_cache():
    """Delete cache entries past TTL. Runs hourly via Celery beat.

    Deletes in BATCHES rather than one statement. A single unbounded DELETE takes
    a long transaction and holds locks that block inserts on the hot path -- the
    cleanup job would cause the outage it exists to prevent.
    """
    return _run(_prune())


async def _prune(batch: int = 5_000) -> int:
    from .config import CACHE_TTL_HOURS
    total = 0
    while True:
        # execute() returns a status string -- "DELETE 4231" -- so we learn how
        # many rows actually went. fetchval() would only tell us whether ANY did,
        # and the count would be a guess.
        status = await db.pool().execute(
            """
            DELETE FROM cache_entries
            WHERE id IN (
                SELECT id FROM cache_entries
                WHERE created_at < now() - ($1 || ' hours')::interval
                LIMIT $2
            )
            """,
            str(CACHE_TTL_HOURS), batch,
        )
        deleted = int(status.rsplit(" ", 1)[-1])
        total += deleted
        if deleted < batch:              # partial batch means we caught up
            break
        if total >= 1_000_000:           # safety valve, resume next hour
            break
    return total


@celery_app.task(name="record_usage", max_retries=5,
                 autoretry_for=(Exception,), retry_backoff=True)
def record_usage(request_id: str, tenant_id: int, cache_hit: bool,
                 provider: str | None, model: str | None, prompt_tokens: int,
                 completion_tokens: int, cost_usd: str, latency_ms: int):
    # Retries are safe: usage_events.request_id is UNIQUE and the insert is
    # ON CONFLICT DO NOTHING (§22).
    _run(accounting.record(
        uuid.UUID(request_id), tenant_id, cache_hit, provider, model,
        Usage(prompt_tokens, completion_tokens), Decimal(cost_usd), latency_ms,
    ))
