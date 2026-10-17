"""The semantic cache and its four guards (DESIGN.md §3).

Guard 1 (structural): namespace = hash(model, system prompt, temperature bucket).
    Similarity is only ever compared INSIDE one (tenant, namespace) bucket, so a
    cross-tenant or cross-model hit is impossible by construction rather than by
    threshold. Most of the safety lives here, not in the number.

Guard 2 (structural): eligibility rules -- is_eligible().

Guard 3: lexical_conflict(). MEASURED NECESSITY, not belt-and-braces. The
    adversarial and paraphrase similarity distributions OVERLAP: "Transfer funds
    from savings to checking" vs the reverse scores 0.992, while the legitimate
    paraphrase "capital of France" / "capital city of France" scores 0.957. No
    threshold separates them. Cosine gives recall; this gives precision.

Guard 4: SIMILARITY_THRESHOLD from bench/threshold_sweep.py, plus the adversarial
    regression set in tests/test_cache_correctness.py.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import numpy as np

from .. import db
from ..config import (CACHE_TTL_HOURS, MAX_CACHEABLE_PROMPT_CHARS,
                      MAX_CACHEABLE_TEMPERATURE, SIMILARITY_THRESHOLD)


@dataclass(frozen=True)
class CacheHit:
    response: str
    similarity: float
    prompt_tokens: int
    completion_tokens: int
    model: str
    provider: str


def namespace(model: str, messages: list[dict], temperature: float) -> str:
    """Bucket key. Temperature is BUCKETED, not exact: 0.0 and 0.05 produce
    interchangeable output, and a distinct bucket per float would give every
    request its own cache of one."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system")
    bucket = round(temperature * 10) / 10
    raw = f"{model}\x00{bucket}\x00{system}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


def query_text(messages: list[dict]) -> str:
    """Only non-system turns are embedded -- the system prompt is already part of
    the namespace, so including it would add a large constant to every vector and
    compress the similarity range that actually discriminates."""
    return "\n".join(m["content"] for m in messages if m["role"] != "system")


# ------------------------------------------------------- lexical guard ----
# Embeddings measure TOPICAL similarity. Negation and argument order are
# STRUCTURAL properties that survive embedding almost untouched -- "not"
# contributes little to a sentence vector, and the models are largely
# order-insensitive. So the guard inspects exactly what the embedding discards.

_POLARITY = {
    "not", "no", "never", "cannot", "cant", "dont", "doesnt", "isnt", "wasnt",
    "without", "except", "unless", "neither", "nor",
}
_ANTONYMS = [
    {"safe", "unsafe"}, {"start", "stop"}, {"largest", "smallest"},
    {"read", "write"}, {"staging", "production"}, {"enable", "disable"},
    {"succeed", "fail"}, {"open", "close"}, {"add", "remove"},
    {"grant", "revoke"}, {"min", "max"}, {"before", "after"},
    {"increase", "decrease"}, {"asc", "desc"}, {"allow", "deny"},
]
_NEG_PREFIXES = ("un", "non", "in", "dis", "im")
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return _TOKEN.findall(s.lower())


def lexical_conflict(a: str, b: str) -> str | None:
    """Reason to REFUSE an otherwise-similar cache hit, or None to allow it.

    ~20 us of string work, three orders of magnitude below the network call it
    protects. Measured effect at threshold 0.94: wrong hits 2/10 -> 0/10, with
    useful hits unchanged at 6/6.
    """
    ta, tb = _tokens(a), _tokens(b)

    # 1. Same bag of words, different order. "USD to EUR" and "EUR to USD" are
    #    different questions with identical multisets -- invisible to cosine.
    if sorted(ta) == sorted(tb):
        return None if ta == tb else "argument order differs"

    sa, sb = set(ta), set(tb)
    diff = sa ^ sb

    # 2. An explicit polarity marker on exactly one side.
    if diff & _POLARITY:
        return f"polarity token {sorted(diff & _POLARITY)}"

    # 3. Morphological negation: 'safe' on one side, 'unsafe' on the other.
    for w in diff:
        for p in _NEG_PREFIXES:
            if w.startswith(p) and w[len(p):] in (sa | sb) - {w}:
                return f"negated form {w!r}"

    # 4. A known antonym pair split across the two sides.
    for pair in _ANTONYMS:
        if len(pair & diff) == 2:
            return f"antonym pair {sorted(pair)}"

    return None


def is_eligible(temperature: float, prompt: str, has_tools: bool = False) -> bool:
    """A request asking for variation is not cacheable. Returning a stored answer
    for temperature=1.2 is not a cache -- it is a bug that looks like a cache."""
    if temperature > MAX_CACHEABLE_TEMPERATURE:
        return False
    if has_tools:
        return False
    return 0 < len(prompt) <= MAX_CACHEABLE_PROMPT_CHARS


async def lookup(tenant_id: int, ns: str, query: str, vector: np.ndarray,
                 threshold: float = SIMILARITY_THRESHOLD) -> CacheHit | None:
    """Exact KNN inside one bucket. See DESIGN.md §10 for why this is not an
    HNSW/ANN search -- post-filtering would silently return zero rows."""
    row = await db.pool().fetchrow(
        """
        SELECT prompt, response, prompt_tokens, completion_tokens, model, provider,
               1 - (embedding <=> $3) AS similarity
        FROM cache_entries
        WHERE tenant_id = $1
          AND namespace = $2
          AND created_at > now() - ($4 || ' hours')::interval
        ORDER BY embedding <=> $3
        LIMIT 1
        """,
        tenant_id, ns, vector, str(CACHE_TTL_HOURS),
    )
    if row is None or row["similarity"] < threshold:
        return None

    # Guard 3. Cosine said "close enough"; this asks whether they are the same
    # QUESTION. Without it, a 0.992 argument transposition is served as a hit.
    if lexical_conflict(query, row["prompt"]):
        return None

    return CacheHit(
        response=row["response"],
        similarity=float(row["similarity"]),
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        model=row["model"],
        provider=row["provider"],
    )


async def store(tenant_id: int, ns: str, prompt: str, vector: np.ndarray,
                response: str, prompt_tokens: int, completion_tokens: int,
                provider: str, model: str) -> None:
    await db.pool().execute(
        """
        INSERT INTO cache_entries
            (tenant_id, namespace, prompt, embedding, response,
             prompt_tokens, completion_tokens, provider, model)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        tenant_id, ns, prompt, vector, response,
        prompt_tokens, completion_tokens, provider, model,
    )
