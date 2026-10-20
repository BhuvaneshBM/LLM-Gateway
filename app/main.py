"""The request path.

Read this top to bottom and the whole design is visible: every step before the
provider call costs about 16 ms in total, and everything that does not have to
block the caller happens after the response is already on its way back.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from . import accounting, breaker, db, limits, tasks, telemetry
from .auth import Tenant, resolve
from .cache import embed, semantic
from .config import DEFAULT_MODEL, PROVIDER_CHAIN
from .models import ChatRequest, ChatResponse
from .providers.base import Usage, cost_usd
from .providers.gemini import GeminiProvider
from .providers.groq import GroqProvider
from .providers.mock import MockProvider

_REGISTRY = {"mock": MockProvider, "groq": GroqProvider, "gemini": GeminiProvider}
CHAIN = [_REGISTRY[name]() for name in PROVIDER_CHAIN if name in _REGISTRY]
log = logging.getLogger("gateway")

@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.connect()
    await limits.connect()
    # Eager load: a lazy first encode costs 2-3 s. Paying it here means the
    # container is not healthy until it can actually serve, instead of
    # accepting traffic and putting 3 s into someone's p99.
    await run_in_threadpool(embed.load)
    yield
    await db.close()
    await limits.close()


app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    await db.pool().fetchval("SELECT 1")
    await limits.client().ping()
    return {"ok": True, "providers": [p.name for p in CHAIN]}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, tenant: Tenant = Depends(resolve)) -> ChatResponse:
    t0 = time.perf_counter()
    request_id = uuid.uuid4()
    model = req.model or DEFAULT_MODEL
    messages = [m.model_dump() for m in req.messages]

    # -- 1. rate limit ------------------------------------------------ ~1 ms
    allowed, remaining = await limits.allow_request(
        tenant.id, tenant.rate_per_sec, tenant.burst)
    if not allowed:
        telemetry.REQUESTS.labels(tenant.name, "rate_limited").inc()
        raise HTTPException(429, "rate limit exceeded",
                            headers={"Retry-After": "1"})

    # -- 2. budget pre-authorisation ---------------------------------- ~1 ms
    estimate = accounting.estimate(model, messages, req.max_tokens)
    within, spent = await limits.reserve_budget(
        tenant.id, tenant.monthly_budget_usd, estimate)
    if not within:
        telemetry.REQUESTS.labels(tenant.name, "over_budget").inc()
        # 402 rather than 429: retrying will not help, the money is gone.
        raise HTTPException(402, f"monthly budget exhausted (spent ${spent})")

    # -- 3. semantic cache -------------------------------------------- ~13 ms
    prompt = semantic.query_text(messages)
    vector = None
    hit = None
    eligible = not req.no_cache and semantic.is_eligible(req.temperature, prompt)

    if eligible:
        with telemetry.STAGE.labels("embed").time():
            # encode() is CPU-bound and BLOCKING. Calling it directly on the
            # event loop stalls every other in-flight request for its duration --
            # 5 ms x 200 concurrent requests serialises into a 1 s queue. The
            # threadpool keeps the loop free.
            vector = await run_in_threadpool(embed.encode, prompt)
        ns = semantic.namespace(model, messages, req.temperature)
        with telemetry.STAGE.labels("cache_lookup").time():
            hit = await semantic.lookup(tenant.id, ns, prompt, vector)
    else:
        telemetry.CACHE.labels("skip").inc()

    if hit is not None:
        await limits.refund_budget(tenant.id, estimate)   # a hit costs nothing
        latency_ms = int((time.perf_counter() - t0) * 1000)
        telemetry.CACHE.labels("hit").inc()
        telemetry.REQUESTS.labels(tenant.name, "cached").inc()
        telemetry.LATENCY.labels("true").observe(latency_ms / 1000)
        try:
            result = tasks.record_usage.delay(
                str(request_id),
                tenant.id,
                True,
                hit.provider,
                hit.model,
                hit.prompt_tokens,
                hit.completion_tokens,
                "0",
                latency_ms,
            )
            log.warning("CACHE HIT: record_usage queued %s", result.id)
        except Exception:
            log.exception("CACHE HIT: record_usage enqueue failed")
        return ChatResponse(
            id=str(request_id), content=hit.response, model=hit.model,
            provider=hit.provider, cached=True, similarity=hit.similarity,
            prompt_tokens=hit.prompt_tokens,
            completion_tokens=hit.completion_tokens,
            cost_usd=0.0, latency_ms=latency_ms)

    if eligible:
        telemetry.CACHE.labels("miss").inc()
        log.warning("CACHE miss counter incremented")

    # -- 4. single flight ---------------------------------------------------
    # Without this, N identical concurrent requests all miss and all pay. The
    # first takes the lock and calls the provider; the rest wait for its result
    # for at most FILL_WAIT_BUDGET, then give up and call anyway.
    fill_key = None
    if eligible:
        fill_key = hashlib.sha256(
            f"{semantic.namespace(model, messages, req.temperature)}\x00{prompt}".encode()
        ).hexdigest()[:32]
        if not await limits.acquire_fill_lock(fill_key):
            payload = await limits.await_fill_result(fill_key)
            if payload is not None:
                shared = json.loads(payload)
                await limits.refund_budget(tenant.id, estimate)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                telemetry.CACHE.labels("coalesced").inc()
                telemetry.LATENCY.labels("true").observe(latency_ms / 1000)
                return ChatResponse(
                    id=str(request_id), content=shared["text"],
                    model=shared["model"], provider=shared["provider"],
                    cached=True, similarity=1.0,
                    prompt_tokens=shared["prompt_tokens"],
                    completion_tokens=shared["completion_tokens"],
                    cost_usd=0.0, latency_ms=latency_ms)
            fill_key = None          # holder died or was too slow; fill it ourselves

    # -- 5. the provider ---------------------------------------------- ~900 ms
    try:
        with telemetry.STAGE.labels("provider").time():
            completion, tried = await breaker.call_with_failover(
                CHAIN, messages, model, req.temperature, req.max_tokens)
    except breaker.AllProvidersDown as e:
        if fill_key:
            # Release, or every waiter blocks for the full lock TTL on a
            # request that already failed.
            await limits.release_fill_lock(fill_key)
        await limits.refund_budget(tenant.id, estimate)
        telemetry.REQUESTS.labels(tenant.name, "upstream_down").inc()
        raise HTTPException(503, f"all providers unavailable: {e}") from e

    if fill_key:
        await limits.publish_fill_result(fill_key, json.dumps({
            "text": completion.text, "model": completion.model,
            "provider": completion.provider,
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
        }))

    for name in tried:
        telemetry.PROVIDER_CALLS.labels(name, "attempt").inc()

    # -- 6. settle and reply ------------------------------------------------
    actual = cost_usd(completion.model, completion.usage)
    await limits.settle_budget(tenant.id, estimate, actual)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    telemetry.REQUESTS.labels(tenant.name, "ok").inc()
    log.warning("REQUESTS counter incremented")
    telemetry.LATENCY.labels("false").observe(latency_ms / 1000)
    log.warning("LATENCY histogram observed")
    telemetry.COST.labels(tenant.name, completion.model).inc(float(actual))

    # -- 7. off the hot path ------------------------------------------------
    if eligible and vector is not None:
        try:
            result = tasks.persist_cache_entry.delay(
                tenant.id,
                semantic.namespace(model, messages, req.temperature),
                prompt,
                vector.tolist(),
                completion.text,
                completion.usage.prompt_tokens,
                completion.usage.completion_tokens,
                completion.provider,
                completion.model,
            )
            log.warning("CACHE STORE queued %s", result.id)
        except Exception:
            log.exception("CACHE STORE enqueue failed")

    tasks.record_usage.delay(
        str(request_id), tenant.id, False, completion.provider, completion.model,
        completion.usage.prompt_tokens, completion.usage.completion_tokens,
        str(actual), latency_ms)

    return ChatResponse(
        id=str(request_id), content=completion.text, model=completion.model,
        provider=completion.provider, cached=False, similarity=None,
        prompt_tokens=completion.usage.prompt_tokens,
        completion_tokens=completion.usage.completion_tokens,
        cost_usd=float(actual), latency_ms=latency_ms)


@app.get("/v1/usage")
async def usage(tenant: Tenant = Depends(resolve)):
    """What every finance team actually asks for: who spent what."""
    spent = await limits.spend_so_far(tenant.id)
    rows = await db.pool().fetch(
        """
        SELECT date_trunc('hour', created_at) AS hour,
               count(*) AS requests,
               count(*) FILTER (WHERE cache_hit) AS cache_hits,
               sum(cost_usd) AS cost_usd
        FROM usage_events
        WHERE tenant_id = $1 AND created_at > now() - interval '24 hours'
        GROUP BY 1 ORDER BY 1 DESC
        """, tenant.id)
    return {
        "tenant": tenant.name,
        "month_to_date_usd": float(spent),
        "budget_usd": float(tenant.monthly_budget_usd),
        "hourly": [dict(r) for r in rows],
    }
