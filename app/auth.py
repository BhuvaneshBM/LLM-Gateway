"""Tenant resolution from an API key."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal

from fastapi import Header, HTTPException

from . import db


@dataclass(frozen=True)
class Tenant:
    id: int
    name: str
    rate_per_sec: float
    burst: int
    monthly_budget_usd: Decimal


async def resolve(authorization: str = Header(...)) -> Tenant:
    """Keys are stored as SHA-256, never in plaintext.

    No bcrypt here on purpose: bcrypt is deliberately slow (~100 ms), which is
    correct for user passwords and wrong for a header checked on every request --
    it would dominate the entire pre-provider budget. API keys are
    high-entropy random strings, not human-chosen, so they are not brute-forceable
    the way a password is and a fast hash is the right tool.
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "expected 'Authorization: Bearer <key>'")
    digest = hashlib.sha256(authorization[7:].encode()).hexdigest()

    row = await db.pool().fetchrow(
        "SELECT id, name, rate_per_sec, burst, monthly_budget_usd "
        "FROM tenants WHERE api_key_hash = $1", digest,
    )
    if row is None:
        raise HTTPException(401, "unknown API key")
    return Tenant(id=row["id"], name=row["name"],
                  rate_per_sec=row["rate_per_sec"], burst=row["burst"],
                  monthly_budget_usd=row["monthly_budget_usd"])
