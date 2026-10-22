import pytest

from app import breaker, limits
from app.config import BREAKER_FAILURE_THRESHOLD
from app.providers.base import Completion, ProviderError, Usage
from app.providers.mock import MockProvider


class AlwaysFails:
    name = "always_fails"

    def __init__(self, retryable=True):
        self.retryable = retryable
        self.calls = 0

    async def complete(self, *a, **k):
        self.calls += 1
        raise ProviderError(self.name, "boom", retryable=self.retryable)


MSGS = [{"role": "user", "content": "hi"}]


async def test_breaker_opens_after_threshold(redis_client):
    p = AlwaysFails()
    for _ in range(BREAKER_FAILURE_THRESHOLD):
        with pytest.raises(breaker.AllProvidersDown):
            await breaker.call_with_failover([p], MSGS, "mock-small", 0.0, 16)
    assert await breaker.is_open(p.name)


async def test_open_circuit_short_circuits_without_calling(redis_client):
    p = AlwaysFails()
    for _ in range(BREAKER_FAILURE_THRESHOLD):
        with pytest.raises(breaker.AllProvidersDown):
            await breaker.call_with_failover([p], MSGS, "mock-small", 0.0, 16)
    before = p.calls
    with pytest.raises(breaker.AllProvidersDown):
        await breaker.call_with_failover([p], MSGS, "mock-small", 0.0, 16)
    assert p.calls == before, "an open circuit still called the provider"


async def test_failover_reaches_the_second_provider(redis_client):
    completion, tried = await breaker.call_with_failover(
        [AlwaysFails(), MockProvider()], MSGS, "mock-small", 0.0, 16)
    assert isinstance(completion, Completion)
    assert tried == ["always_fails", "mock"]


async def test_our_own_bad_request_does_not_open_the_circuit(redis_client):
    """A 4xx means WE sent something malformed. Opening the circuit on a healthy
    provider because of our bug would turn a 400 into an outage."""
    p = AlwaysFails(retryable=False)
    with pytest.raises(ProviderError):
        await breaker.call_with_failover([p], MSGS, "mock-small", 0.0, 16)
    assert not await breaker.is_open(p.name)


async def test_success_clears_prior_failures(redis_client):
    p = AlwaysFails()
    for _ in range(BREAKER_FAILURE_THRESHOLD - 1):
        with pytest.raises(breaker.AllProvidersDown):
            await breaker.call_with_failover([p], MSGS, "mock-small", 0.0, 16)
    await breaker.call_with_failover([MockProvider()], MSGS, "mock-small", 0.0, 16)
    assert not await breaker.is_open("mock")
