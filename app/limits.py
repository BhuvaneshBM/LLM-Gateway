"""Rate limiting and budgets. Both are Lua scripts, and that is the point.

GET the bucket, compute in Python, SET it back is a RACE. Two concurrent
requests from the same tenant read the same token count, both decide they have
room, and both proceed -- so a 10 rps limit passes 20. Redis runs a Lua script
atomically against a single-threaded event loop, so the check and the decrement
cannot interleave.

This is the difference between knowing what a token bucket is and having built
one that survives concurrency.
"""

from __future__ import annotations

import time
from decimal import Decimal

import redis.asyncio as aioredis

from .config import REDIS_URL

_redis: aioredis.Redis | None = None

# ---------------------------------------------------------------- scripts --
_TOKEN_BUCKET = """
local rate  = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now   = tonumber(ARGV[3])
local cost  = tonumber(ARGV[4])

local state  = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts     = tonumber(state[2])

if tokens == nil then          -- first request for this tenant
  tokens = burst
  ts = now
end

local elapsed = now - ts
if elapsed < 0 then elapsed = 0 end
tokens = math.min(burst, tokens + elapsed * rate)

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
-- Expire after a full refill plus slack, so idle tenants do not leak keys.
redis.call('EXPIRE', KEYS[1], math.ceil(burst / rate) + 60)

return {allowed, tostring(tokens)}
"""

# Pre-authorise against an ESTIMATE, then settle with the actual cost once the
# provider replies -- the same authorise/capture split a card network uses. You
# cannot know the true cost before the call, and refusing to cap spend until
# after you have spent it is not a cap.
_RESERVE = """
local budget = tonumber(ARGV[1])
local amount = tonumber(ARGV[2])
local ttl    = tonumber(ARGV[3])

local spent = tonumber(redis.call('GET', KEYS[1]) or '0')
if spent >= budget then
  return {0, tostring(spent)}
end

local updated = spent + amount
redis.call('SET', KEYS[1], tostring(updated), 'EX', ttl)
return {1, tostring(updated)}
"""

_ADJUST = """
local delta = tonumber(ARGV[1])
local ttl   = tonumber(ARGV[2])
local spent = tonumber(redis.call('GET', KEYS[1]) or '0') + delta
if spent < 0 then spent = 0 end
redis.call('SET', KEYS[1], tostring(spent), 'EX', ttl)
return tostring(spent)
"""

_MONTH_TTL = 40 * 24 * 3600      # comfortably longer than any calendar month

_bucket_sha = _reserve_sha = _adjust_sha = None


async def connect() -> aioredis.Redis:
    global _redis, _bucket_sha, _reserve_sha, _adjust_sha
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        _bucket_sha = await _redis.script_load(_TOKEN_BUCKET)
        _reserve_sha = await _redis.script_load(_RESERVE)
        _adjust_sha = await _redis.script_load(_ADJUST)
    return _redis


async def close() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def client() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("limits.connect() has not run")
    return _redis


def _budget_key(tenant_id: int) -> str:
    return f"budget:{tenant_id}:{time.strftime('%Y-%m')}"


# ------------------------------------------------------------------- api ---
async def allow_request(tenant_id: int, rate: float, burst: int) -> tuple[bool, float]:
    allowed, remaining = await client().evalsha(
        _bucket_sha, 1, f"rl:{tenant_id}", rate, burst, time.time(), 1
    )
    return bool(allowed), float(remaining)


async def reserve_budget(tenant_id: int, budget: Decimal,
                         estimate: Decimal) -> tuple[bool, Decimal]:
    ok, spent = await client().evalsha(
        _reserve_sha, 1, _budget_key(tenant_id),
        str(budget), str(estimate), _MONTH_TTL,
    )
    return bool(ok), Decimal(spent)


async def settle_budget(tenant_id: int, estimate: Decimal, actual: Decimal) -> None:
    """Replace the estimate with the truth. Usually a refund, since max_tokens
    is an upper bound and completions are shorter than it."""
    delta = actual - estimate
    if delta != 0:
        await client().evalsha(_adjust_sha, 1, _budget_key(tenant_id),
                               str(delta), _MONTH_TTL)


async def refund_budget(tenant_id: int, estimate: Decimal) -> None:
    """A cache hit costs nothing, so the whole pre-authorisation goes back."""
    await client().evalsha(_adjust_sha, 1, _budget_key(tenant_id),
                           str(-estimate), _MONTH_TTL)


async def spend_so_far(tenant_id: int) -> Decimal:
    return Decimal(await client().get(_budget_key(tenant_id)) or "0")


# ------------------------------------------------------- single flight ----
# Cache stampede: 100 identical requests arriving on a cold entry all miss, and
# all 100 call the provider. Because the cache write is asynchronous (§25) the
# window is WIDER here than in a normal cache -- it stays open until Celery
# lands the row, not just until the first response returns.
#
# Fix: the first request to arrive takes a lock and calls the provider; the rest
# wait for it to publish the result. The holder publishes to Redis directly
# rather than waiting on the Celery write, so the two paths stay decoupled.

FILL_LOCK_TTL = 30          # seconds; must exceed provider timeout
FILL_RESULT_TTL = 60
FILL_POLL_INTERVAL = 0.05
FILL_WAIT_BUDGET = 2.0      # cap: never wait longer than a normal miss


async def acquire_fill_lock(key: str) -> bool:
    """SET NX is the whole lock. Returns True if WE are the filler."""
    return bool(await client().set(f"fill:{key}", "1", nx=True, ex=FILL_LOCK_TTL))


async def publish_fill_result(key: str, payload: str) -> None:
    r = client()
    await r.set(f"fillres:{key}", payload, ex=FILL_RESULT_TTL)
    await r.delete(f"fill:{key}")


async def release_fill_lock(key: str) -> None:
    """Called when the fill FAILED. Without this the lock lingers for its TTL
    and every waiter blocks on a request that is never coming."""
    await client().delete(f"fill:{key}")


async def await_fill_result(key: str) -> str | None:
    """Poll for the holder's result. Returns None on timeout, and the caller
    then proceeds to call the provider itself -- a duplicate call is far better
    than an unbounded wait on a holder that may have died."""
    import asyncio
    deadline = time.monotonic() + FILL_WAIT_BUDGET
    while time.monotonic() < deadline:
        payload = await client().get(f"fillres:{key}")
        if payload is not None:
            return payload
        await asyncio.sleep(FILL_POLL_INTERVAL)
    return None
