"""THE test. A semantic cache can return a confidently WRONG answer, and this is
the only thing standing between the project and that bug.

Every similarity below is MEASURED with all-MiniLM-L6-v2, not assumed. The
headline finding (DESIGN.md §3) is that the two classes OVERLAP -- an argument
transposition scores 0.992 while a legitimate paraphrase scores 0.957 -- so the
threshold alone cannot separate them and the lexical guard is load-bearing
rather than defensive.

Needs no database and no Redis: it tests the decision where it actually lives.
"""

import numpy as np
import pytest

from app.cache.embed import encode
from app.cache.semantic import is_eligible, lexical_conflict, namespace
from app.config import SIMILARITY_THRESHOLD

# Pairs that MUST NOT be served from one another's cache entry.
# The comment on each is its measured cosine similarity.
ADVERSARIAL = [
    ("What is the capital of France?", "What is the capital of Germany?"),          # 0.663
    ("How do I start the payment service?", "How do I stop the payment service?"),  # 0.725
    ("Show me the 10 largest accounts", "Show me the 10 smallest accounts"),        # 0.824
    ("Delete the staging database", "Delete the production database"),              # 0.791
    ("Grant read access to Alice", "Grant write access to Alice"),                  # 0.890
    ("What did revenue do in Q1?", "What did revenue do in Q3?"),                   # 0.918
    ("Did the payment succeed?", "Did the payment not succeed?"),                   # 0.935
    ("Is this transaction safe?", "Is this transaction unsafe?"),                   # 0.965 <-- above threshold
    ("Convert 100 USD to EUR", "Convert 100 EUR to USD"),                           # 0.980 <-- above threshold
    ("Transfer funds from savings to checking",
     "Transfer funds from checking to savings"),                                    # 0.992 <-- above threshold
]

# Pairs that SHOULD hit. Without these the suite is satisfied by threshold=1.0,
# which is perfectly safe and perfectly useless.
PARAPHRASE = [
    ("What is the capital of France?", "What's the capital city of France?"),       # 0.957
    ("List all active users", "List every active user"),                            # 0.956
    ("Explain how OAuth works", "Explain how OAuth works."),                        # 0.988
    ("How do I reset my password?", "How can I reset my password?"),                # 0.992
    ("What is the refund policy?", "What's the refund policy?"),                    # 0.995
    ("How do I cancel my subscription?", "How can I cancel my subscription?"),      # 0.996
]


def cosine(a: str, b: str) -> float:
    return float(np.dot(encode(a), encode(b)))      # encode() is L2-normalised


def would_hit(a: str, b: str) -> bool:
    """The full decision, exactly as app/cache/semantic.py makes it."""
    if cosine(a, b) < SIMILARITY_THRESHOLD:
        return False
    return lexical_conflict(a, b) is None


# ---------------------------------------------------------- the real test --
@pytest.mark.parametrize("a,b", ADVERSARIAL)
def test_adversarial_pairs_never_hit(a, b):
    assert not would_hit(a, b), (
        f"sim={cosine(a, b):.4f} and no lexical conflict detected: the cache "
        f"would answer\n  {b!r}\nwith the stored response for\n  {a!r}"
    )


@pytest.mark.parametrize("a,b", PARAPHRASE)
def test_paraphrases_do_hit(a, b):
    assert would_hit(a, b), (
        f"sim={cosine(a, b):.4f}, conflict={lexical_conflict(a, b)!r}: the guard "
        f"or the threshold is rejecting a legitimate paraphrase"
    )


def test_threshold_alone_is_insufficient():
    """Pins the project's central FINDING (DESIGN.md §3).

    If this ever passes-by-becoming-false -- i.e. the distributions separate --
    then a future embedding model has made the lexical guard unnecessary, and
    that is worth knowing rather than silently carrying dead code.
    """
    worst_ok = min(cosine(a, b) for a, b in PARAPHRASE)
    best_bad = max(cosine(a, b) for a, b in ADVERSARIAL)
    assert best_bad > worst_ok, (
        f"distributions no longer overlap (adversarial max {best_bad:.4f} <= "
        f"paraphrase min {worst_ok:.4f}); re-evaluate whether the guard is needed"
    )


def test_guard_is_what_saves_the_high_similarity_pairs():
    """The three pairs above threshold are caught by the guard, not the number."""
    above = [(a, b) for a, b in ADVERSARIAL if cosine(a, b) >= SIMILARITY_THRESHOLD]
    assert len(above) >= 3, "expected the known high-similarity adversarial pairs"
    for a, b in above:
        assert lexical_conflict(a, b) is not None, f"guard missed {a!r} vs {b!r}"


# ------------------------------------------------------- the guard itself --
@pytest.mark.parametrize("a,b,expected", [
    ("Convert 100 USD to EUR", "Convert 100 EUR to USD", "argument order differs"),
    ("Is this transaction safe?", "Is this transaction unsafe?", "negated form"),
    ("Did the payment succeed?", "Did the payment not succeed?", "polarity token"),
    ("Grant read access to Alice", "Grant write access to Alice", "antonym pair"),
])
def test_guard_reasons(a, b, expected):
    reason = lexical_conflict(a, b)
    assert reason is not None and expected in reason, f"got {reason!r}"


def test_guard_allows_genuine_paraphrase():
    assert lexical_conflict("What is the capital of France?",
                            "What's the capital city of France?") is None
    assert lexical_conflict("same text", "same text") is None


# ---- the structural guards (DESIGN.md §3, guards 1 and 2) ----
def test_namespace_separates_models_prompts_and_temperature():
    m = [{"role": "system", "content": "You are terse."},
         {"role": "user", "content": "hello"}]
    base = namespace("model-a", m, 0.0)
    assert namespace("model-b", m, 0.0) != base           # different model
    assert namespace("model-a", [{"role": "system", "content": "You are verbose."},
                                 m[1]], 0.0) != base      # different system prompt
    assert namespace("model-a", m, 0.9) != base           # different temp bucket
    assert namespace("model-a", m, 0.04) == base          # same bucket -> same ns


def test_high_temperature_is_not_cacheable():
    assert is_eligible(0.0, "hi")
    assert not is_eligible(1.2, "hi")
    assert not is_eligible(0.0, "hi", has_tools=True)
    assert not is_eligible(0.0, "")
