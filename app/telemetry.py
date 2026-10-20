"""Prometheus metrics. These are the numbers the README quotes."""

from prometheus_client import Counter, Histogram

REQUESTS = Counter("gw_requests_total", "Requests", ["tenant", "outcome"])
CACHE = Counter("gw_cache_total", "Cache lookups", ["result"])   # hit | miss | skip
PROVIDER_CALLS = Counter("gw_provider_calls_total", "Provider calls",
                         ["provider", "outcome"])
COST = Counter("gw_cost_usd_total", "Cost in USD", ["tenant", "model"])

# Buckets are chosen for THIS system, not left at the library default. A cache
# hit lands near 16 ms and a provider call near 900 ms; the default buckets put
# both in one giant bin and the histogram tells you nothing.
LATENCY = Histogram(
    "gw_request_latency_seconds", "End-to-end latency", ["cached"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
)
STAGE = Histogram(
    "gw_stage_latency_seconds", "Per-stage latency", ["stage"],
    buckets=(0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0, 5.0),
)
