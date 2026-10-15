"""Deterministic provider with a shaped latency distribution.

EVERY load test uses this, and that is a methodology decision rather than a
shortcut (DESIGN.md §5c). Benchmarking against a real provider measures THEIR
latency variance and THEIR rate limits; you would be publishing someone else's
p99. Injecting a known distribution isolates the gateway's own overhead, costs
nothing, and reproduces exactly.

Latency is lognormal because real inference latency is right-skewed -- a long
tail of slow responses, never a negative one. A uniform distribution would make
the p99 look far better than reality.
"""

from __future__ import annotations

import asyncio
import hashlib
import random

from ..config import (MOCK_FAILURE_RATE, MOCK_LATENCY_MEAN_MS,
                      MOCK_LATENCY_SIGMA)
from .base import Completion, ProviderError, Usage


class MockProvider:
    name = "mock"

    def __init__(self, failure_rate: float | None = None,
                 mean_ms: float | None = None):
        self.failure_rate = MOCK_FAILURE_RATE if failure_rate is None else failure_rate
        self.mean_ms = MOCK_LATENCY_MEAN_MS if mean_ms is None else mean_ms
        self._rng = random.Random(0)

    async def complete(self, messages, model, temperature, max_tokens) -> Completion:
        if self._rng.random() < self.failure_rate:
            raise ProviderError(self.name, "injected failure")

        mu = float(__import__("math").log(max(self.mean_ms, 1.0))) - (MOCK_LATENCY_SIGMA ** 2) / 2
        delay_ms = self._rng.lognormvariate(mu, MOCK_LATENCY_SIGMA)
        await asyncio.sleep(min(delay_ms, 30_000) / 1000.0)

        prompt = "\n".join(m["content"] for m in messages)
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        text = f"[mock:{digest}] {prompt[:120]}"

        return Completion(
            text=text,
            usage=Usage(prompt_tokens=max(len(prompt) // 4, 1),
                        completion_tokens=max(len(text) // 4, 1)),
            provider=self.name,
            model=model,
        )
