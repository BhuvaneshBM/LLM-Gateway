"""Circuit breaker with a provider failover chain.

State lives in REDIS, not in the process. With three API replicas an in-process
breaker means each one has to independently discover the provider is down --
three times the failed requests, and three different opinions about whether the
dependency is healthy. Shared state means the first replica to notice protects
the rest.

The state machine is deliberately simple:

    CLOSED     normal. Consecutive failures accumulate.
    OPEN       >= threshold failures. Every call short-circuits until cooldown.
    HALF-OPEN  cooldown elapsed. The next call is allowed through as a probe;
               success closes the circuit, failure re-opens it.

HALF-OPEN is implicit rather than a stored state: once `open_until` has passed,
`allow()` returns True again, so the next caller IS the probe. Under high
concurrency several probes may slip through together. That is a deliberate
trade -- a strictly single-probe implementation needs another lock, and letting
three requests test a recovering provider is cheap.
"""

from __future__ import annotations

import time

from . import limits
from .config import BREAKER_COOLDOWN_SECONDS, BREAKER_FAILURE_THRESHOLD
from .providers.base import Completion, Provider, ProviderError


def _fail_key(name: str) -> str:
    return f"breaker:{name}:failures"


def _open_key(name: str) -> str:
    return f"breaker:{name}:open_until"


async def is_open(name: str) -> bool:
    raw = await limits.client().get(_open_key(name))
    return raw is not None and time.time() < float(raw)


async def record_success(name: str) -> None:
    r = limits.client()
    await r.delete(_fail_key(name), _open_key(name))


async def record_failure(name: str) -> int:
    r = limits.client()
    failures = await r.incr(_fail_key(name))
    await r.expire(_fail_key(name), BREAKER_COOLDOWN_SECONDS * 4)
    if failures >= BREAKER_FAILURE_THRESHOLD:
        await r.set(_open_key(name), str(time.time() + BREAKER_COOLDOWN_SECONDS),
                    ex=BREAKER_COOLDOWN_SECONDS + 5)
    return failures


class AllProvidersDown(RuntimeError):
    pass


async def call_with_failover(chain: list[Provider], messages: list[dict],
                             model: str, temperature: float,
                             max_tokens: int) -> tuple[Completion, list[str]]:
    """Walk the chain until one succeeds. Returns (completion, providers_tried).

    A non-retryable ProviderError -- a 4xx, meaning WE sent something malformed --
    does not count against the breaker and is raised immediately. Opening the
    circuit on a healthy provider because of our own bad request would turn a
    400 into an outage.
    """
    tried: list[str] = []
    last: Exception | None = None

    for provider in chain:
        if await is_open(provider.name):
            tried.append(f"{provider.name}:open")
            continue
        tried.append(provider.name)
        try:
            completion = await provider.complete(messages, model, temperature, max_tokens)
        except ProviderError as e:
            if not e.retryable:
                raise
            await record_failure(provider.name)
            last = e
            continue
        except Exception as e:                      # timeout, transport, parse
            await record_failure(provider.name)
            last = e
            continue
        await record_success(provider.name)
        return completion, tried

    raise AllProvidersDown(f"chain exhausted {tried}: {last}")
