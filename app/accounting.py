"""Cost estimation and the usage ledger."""

from __future__ import annotations

import uuid
from decimal import Decimal

from . import db
from .providers.base import Usage, cost_usd


def estimate(model: str, messages: list[dict], max_tokens: int) -> Decimal:
    """Upper bound, used to PRE-AUTHORISE against the budget.

    Deliberately pessimistic: assume the completion runs to max_tokens. A tenant
    at their cap should be stopped by the worst case, not sail past it on an
    optimistic guess and overshoot. settle_budget() refunds the difference
    moments later.

    ~4 characters per token is a rough English approximation -- good enough for a
    reservation, never used for the reported number.
    """
    chars = sum(len(m["content"]) for m in messages)
    return cost_usd(model, Usage(prompt_tokens=chars // 4, completion_tokens=max_tokens))


async def record(request_id: uuid.UUID, tenant_id: int, cache_hit: bool,
                 provider: str | None, model: str | None, usage: Usage,
                 cost: Decimal, latency_ms: int) -> None:
    """Append to the usage ledger.

    ON CONFLICT DO NOTHING against the UNIQUE request_id makes this idempotent:
    a Celery retry after a partial failure cannot double-count spend. The
    constraint enforces it, rather than us hoping the task runs exactly once.
    """
    await db.pool().execute(
        """
        INSERT INTO usage_events
            (request_id, tenant_id, cache_hit, provider, model,
             prompt_tokens, completion_tokens, cost_usd, latency_ms)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (request_id) DO NOTHING
        """,
        request_id, tenant_id, cache_hit, provider, model,
        usage.prompt_tokens, usage.completion_tokens, cost, latency_ms,
    )
