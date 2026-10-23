from decimal import Decimal

from app.accounting import estimate
from app.providers.base import Usage, cost_usd


def test_cost_is_exact_decimal_not_float():
    c = cost_usd("mock-small", Usage(prompt_tokens=1_000_000,
                                     completion_tokens=1_000_000))
    assert c == Decimal("2.00")                  # 0.50 + 1.50, exactly
    assert isinstance(c, Decimal)


def test_unknown_model_costs_zero_rather_than_crashing():
    assert cost_usd("model-we-have-never-seen", Usage(100, 100)) == Decimal("0")


def test_estimate_is_pessimistic():
    """The reservation must assume the worst case -- a tenant at their cap should
    be stopped by the upper bound, not sail past on an optimistic guess."""
    msgs = [{"role": "user", "content": "x" * 4_000}]
    est = estimate("mock-small", msgs, max_tokens=1_000)
    actual = cost_usd("mock-small", Usage(prompt_tokens=1_000, completion_tokens=50))
    assert est > actual
