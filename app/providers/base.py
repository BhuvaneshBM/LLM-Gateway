"""The provider seam.

Everything downstream of this file is provider-agnostic, which is what makes
failover a three-line change and load testing free (DESIGN.md §5c).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from ..config import PRICES


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage
    provider: str
    model: str


class ProviderError(RuntimeError):
    """Anything that should trip the circuit breaker: timeout, 5xx, 429."""

    def __init__(self, provider: str, message: str, retryable: bool = True):
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.retryable = retryable


class Provider(Protocol):
    name: str

    async def complete(self, messages: list[dict], model: str,
                       temperature: float, max_tokens: int) -> Completion: ...


def cost_usd(model: str, usage: Usage) -> Decimal:
    """Exact decimal arithmetic. Floats accumulate error over millions of rows,
    and this number ends up in a spend cap."""
    prompt_price, completion_price = PRICES.get(model, (Decimal("0"), Decimal("0")))
    million = Decimal("1000000")
    return (Decimal(usage.prompt_tokens) * prompt_price / million
            + Decimal(usage.completion_tokens) * completion_price / million)
