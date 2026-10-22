"""Proves the Lua script is actually atomic. This is the test that would fail
against a read-modify-write implementation."""

import asyncio
from decimal import Decimal

import pytest

from app import limits


async def test_bucket_allows_exactly_burst_then_refuses(redis_client):
    allowed = 0
    for _ in range(30):
        ok, _ = await limits.allow_request(tenant_id=1, rate=0.001, burst=10)
        allowed += ok
    assert allowed == 10


async def test_concurrent_requests_cannot_exceed_burst(redis_client):
    """THE concurrency test. Read-then-write passes the sequential test above and
    fails this one -- 50 coroutines all read the same token count and all decide
    they have room."""
    results = await asyncio.gather(*[
        limits.allow_request(tenant_id=2, rate=0.001, burst=10) for _ in range(50)
    ])
    assert sum(ok for ok, _ in results) == 10


async def test_bucket_refills_over_time(redis_client):
    for _ in range(5):
        await limits.allow_request(3, rate=100.0, burst=5)
    ok, _ = await limits.allow_request(3, rate=100.0, burst=5)
    assert not ok
    await asyncio.sleep(0.1)                      # 100/s * 0.1 s = 10 tokens
    ok, _ = await limits.allow_request(3, rate=100.0, burst=5)
    assert ok


async def test_budget_reserve_settle_and_refund(redis_client):
    budget = Decimal("1.00")
    ok, spent = await limits.reserve_budget(9, budget, Decimal("0.40"))
    assert ok and spent == Decimal("0.40")

    await limits.settle_budget(9, Decimal("0.40"), Decimal("0.10"))   # refund 0.30
    assert await limits.spend_so_far(9) == Decimal("0.10")

    ok, _ = await limits.reserve_budget(9, budget, Decimal("0.95"))
    assert ok                                     # 0.10 < 1.00, reservation allowed
    ok, _ = await limits.reserve_budget(9, budget, Decimal("0.10"))
    assert not ok                                 # now 1.05 >= 1.00, refused


async def test_cache_hit_refunds_the_whole_reservation(redis_client):
    await limits.reserve_budget(11, Decimal("5"), Decimal("0.25"))
    await limits.refund_budget(11, Decimal("0.25"))
    assert await limits.spend_so_far(11) == Decimal("0")


# ------------------------------------------------------- single flight ----
async def test_only_one_caller_wins_the_fill_lock(redis_client):
    """The stampede guard. 20 coroutines race for the same cold key; exactly one
    may call the provider."""
    got = await asyncio.gather(*[
        limits.acquire_fill_lock("same-key") for _ in range(20)
    ])
    assert sum(got) == 1


async def test_waiter_receives_the_holders_result(redis_client):
    assert await limits.acquire_fill_lock("k1")

    async def holder():
        await asyncio.sleep(0.1)
        await limits.publish_fill_result("k1", '{"text": "shared"}')

    async def waiter():
        return await limits.await_fill_result("k1")

    _, payload = await asyncio.gather(holder(), waiter())
    assert payload is not None and "shared" in payload


async def test_waiter_gives_up_rather_than_hanging(redis_client):
    """If the holder dies, waiters must fall through to calling the provider
    themselves. A duplicate call beats an unbounded wait."""
    assert await limits.acquire_fill_lock("k2")
    limits.FILL_WAIT_BUDGET = 0.2                     # keep the test fast
    assert await limits.await_fill_result("k2") is None


async def test_releasing_the_lock_lets_the_next_caller_fill(redis_client):
    """A failed fill must release, or every waiter blocks for the full lock TTL
    on a request that already failed."""
    assert await limits.acquire_fill_lock("k3")
    assert not await limits.acquire_fill_lock("k3")
    await limits.release_fill_lock("k3")
    assert await limits.acquire_fill_lock("k3")
