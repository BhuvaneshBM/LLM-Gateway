"""Every tunable constant. One place, so nothing is buried in a call site."""

import os
from decimal import Decimal

# --- infrastructure --------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://gw:gw@localhost:5432/gateway")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# --- embedding -------------------------------------------------------------
# Runs IN-PROCESS on CPU. The cache lookup has to be cheaper than the call it
# avoids: ~5 ms local versus ~50 ms and real money for a hosted embedding API.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

# --- semantic cache --------------------------------------------------------
# MEASURED by bench/threshold_sweep.py, not guessed. See DESIGN.md §3 and §35.1.
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.94"))
CACHE_TTL_HOURS = int(os.environ.get("CACHE_TTL_HOURS", "168"))       # 7 days

# Eligibility. A request that asks for variation is not cacheable, and
# pretending otherwise is the bug -- not the threshold.
MAX_CACHEABLE_TEMPERATURE = 0.3
MAX_CACHEABLE_PROMPT_CHARS = 8_000

# --- providers -------------------------------------------------------------
# Ordered failover chain. "mock" is used by EVERY load test (DESIGN.md §5c).
PROVIDER_CHAIN = [p.strip() for p in os.environ.get("PROVIDER_CHAIN", "mock").split(",") if p.strip()]
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("REQUEST_TIMEOUT_SECONDS", "20"))

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# USD per 1M tokens, (prompt, completion). Keep this honest -- every cost number
# the project reports is derived from it.
PRICES: dict[str, tuple[Decimal, Decimal]] = {
    "mock-small":            (Decimal("0.50"), Decimal("1.50")),
    "llama-3.3-70b-versatile": (Decimal("0.59"), Decimal("0.79")),
    "gemini-2.0-flash":      (Decimal("0.10"), Decimal("0.40")),
}
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "mock-small")

# --- circuit breaker -------------------------------------------------------
BREAKER_FAILURE_THRESHOLD = 5      # consecutive failures before opening
BREAKER_COOLDOWN_SECONDS = 30      # open -> half-open after this
BREAKER_HALF_OPEN_PROBES = 1       # successes needed to close again

# --- mock provider (load-test shaping) ------------------------------------
MOCK_LATENCY_MEAN_MS = float(os.environ.get("MOCK_LATENCY_MEAN_MS", "900"))
MOCK_LATENCY_SIGMA = float(os.environ.get("MOCK_LATENCY_SIGMA", "0.35"))
MOCK_FAILURE_RATE = float(os.environ.get("MOCK_FAILURE_RATE", "0.0"))
