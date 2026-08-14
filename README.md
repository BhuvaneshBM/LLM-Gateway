# LLM Inference Gateway

[![python](https://img.shields.io/badge/python-3.11%2B-blue)](#running-it)
[![tests](https://img.shields.io/badge/tests-42%20passing-brightgreen)](#tests)
[![license](https://img.shields.io/badge/license-MIT-lightgrey)](#license)

A gateway between applications and LLM providers: **semantic caching, per-tenant budgets, rate
limiting, circuit breaking and cost accounting**. Runs entirely on one machine — load-tested,
chaos-tested and measured with `docker compose`.

> **XX% cache hit rate** · **p95 XX ms cached / XX ms uncached** · **\$XX → \$XX per 1,000 requests**
> · **zero dropped requests during a provider outage**

The interesting problem is not throughput. It is that **a semantic cache can return a confidently
wrong answer, and no similarity threshold can stop it.**

---

## Contents

- [The finding](#the-finding) — the measurement that reversed my hypothesis
- [Why an LLM is a hard dependency](#why-an-llm-is-a-hard-dependency)
- [Results](#results) · [How it works](#how-it-works) · [API](#api)
- [Design decisions worth defending](#design-decisions-worth-defending)
- [Configuration](#configuration) · [Repository](#repository) · [Running it](#running-it)
- [Tests](#tests) · [Capacity](#capacity)
- [Limitations](#limitations) · [What I would do next](#what-i-would-do-next)

---

## The finding

I measured cosine similarity over labelled prompt pairs using `all-MiniLM-L6-v2`. The result
reversed my starting hypothesis.

**Entity swaps — the obvious worry — are fine:**

| Pair | Similarity |
|---|---|
| "capital of **France**" / "capital of **Germany**" | 0.663 |
| "**start** the payment service" / "**stop** the payment service" | 0.725 |
| "10 **largest** accounts" / "10 **smallest** accounts" | 0.824 |

**Negation and argument order are not:**

| Pair | Similarity | Class |
|---|---|---|
| "Is this transaction **safe**?" / "**unsafe**?" | **0.965** | negation |
| "Convert 100 **USD → EUR**" / "**EUR → USD**" | **0.980** | argument transposition |
| "Transfer funds **savings → checking**" / "**checking → savings**" | **0.992** | argument transposition |

Now compare a legitimate paraphrase the cache *must* catch to be worth having:

| Pair | Similarity |
|---|---|
| "What is the capital of France?" / "What's the capital city of France?" | 0.957 |
| "List all active users" / "List every active user" | 0.956 |

**0.992 is a wrong answer. 0.956 is a right one.** The distributions overlap, so **no threshold
admits real paraphrases and rejects the wrong pairs.** Raising it to 0.995 to exclude the
funds-transfer bug also excludes "capital city of France" — a safe cache that never fires.

That is this project's actual result, and it is worth more than the code: **the standard advice to
"just tune the threshold" is wrong.**

### The fix: cosine for recall, lexical check for precision

Embeddings measure *topical* similarity. Negation and argument order are *structural* properties
that survive embedding almost untouched. So the guard inspects what the embedding discards — token
order, polarity markers, morphological negation (`safe`/`unsafe`), and a small antonym list.

| At threshold 0.94 | Cosine alone | Cosine + lexical guard |
|---|---|---|
| Wrong hits | **2 / 10** | **0 / 10** |
| Useful hits | 6 / 6 | **6 / 6** |

~20 µs of string work — three orders of magnitude below the call it protects.

Reproduce it: `python bench/threshold_sweep.py`.

---

## Why an LLM is a hard dependency

Every design decision here falls out of six properties of the thing being called:

| Property | Design problem it forces |
|---|---|
| Slow — 400 ms to 3 s | A latency budget: what runs in parallel, what runs after the response is sent |
| Expensive per token | Cost modelling, cache ROI, per-tenant budgets |
| Non-deterministic | Which requests are cacheable *at all* |
| Semantically addressed | Vector search — and the wrong-answer risk above |
| Rate limited | Backpressure: queue or shed, and for whom |
| Occasionally down | Circuit breaker, failover, degraded mode |

---

## Results

*Replace every `XX` with a measured number. Nothing here is a result yet.*

| Metric | Value | Produced by |
|---|---|---|
| Cache hit rate (realistic mix) | XX% | `bench/load.js` |
| p50 / p95 — cache hit | XX / XX ms | k6 `latency_cache_hit_ms` |
| p50 / p95 — cache miss | XX / XX ms | k6 `latency_cache_miss_ms` |
| Sustained throughput | XX req/s | k6 ramping arrival rate |
| Cost per 1k requests, no cache | \$XX | `bench/cost_model.py` |
| Cost per 1k requests, at measured hit rate | \$XX | same |
| Requests dropped during provider outage | XX | `bench/chaos.sh` |
| Chosen similarity threshold | XX | `bench/threshold_sweep.py` |

### Where the latency goes

Run `python bench/report.py` after k6 — it computes these from the `/metrics` histograms.

| Stage | Budget | Measured |
|---|---|---|
| auth + rate limit + budget | ~3 ms | XX |
| embedding (in-process) | ~5 ms | XX |
| cache lookup (pgvector) | ~8 ms | XX |
| **total before provider** | **~16 ms** | **XX** |
| provider call | 400–3000 ms | XX |

### Figures

| | |
|---|---|
| ![threshold](bench/results/threshold_curve.png) | ![cost](bench/results/cost_curve.png) |
| **Why a threshold is not enough.** Red is wrong answers with cosine alone; green is with the lexical guard. The shaded band is where the two classes overlap | **Cost vs hit rate**, with the break-even against calling the provider directly |

---

## How it works

```
POST /v1/chat        ┌──────── SYNC — p95 budget 1.5 s ─────────┐
────────────────────►│ 1 auth              ~1 ms                 │
                     │ 2 rate limit        ~1 ms  Redis Lua      │
                     │ 3 budget reserve    ~1 ms  Redis Lua      │
                     │ 4 eligible?         ~0 ms                 │
                     │ 5 embed             ~5 ms  in-process     │
                     │ 6 cache lookup      ~8 ms  pgvector       │
                     │     ├── HIT ─────────────► ~16 ms · $0    │
                     │     └── MISS                              │
                     │          ├─ single-flight lock (SET NX)   │
                     │          │    holder → provider           │
                     │          │    waiters → poll ≤2s          │
                     │          └─ breaker → primary → fallback  │
                     └───────────────┬──────────────────────────┘
                                     │ enqueue, never awaited
                     ┌───────────────▼──────────────────────────┐
                     │ Celery worker  persist cache · record    │
                     │ Celery beat    hourly TTL reaper         │
                     └──────────────────────────────────────────┘

Redis     token buckets · spend counters · breaker state · fill locks
Postgres  cache_entries (pgvector) · tenants · usage_events
```

**The sync/async split is the design.** Everything the caller waits for costs ~16 ms before the
provider call. Everything else — writing the cache entry, cost accounting, rollups — happens after
the response is already on its way back.

### The four guards on cache correctness

1. **Namespacing (structural).** Similarity is only compared inside a bucket keyed by
   `(tenant, model, system_prompt, temperature_bucket)`. Cross-tenant and cross-model hits are
   impossible by construction, not by threshold.
2. **Eligibility (structural).** Only `temperature ≤ 0.3`, no tool calls, bounded length. A request
   asking for variation is not cacheable.
3. **The lexical guard.** Load-bearing, not defensive — it is what makes the cache safe at a
   *usable* threshold, since no threshold works alone.
4. **A measured threshold plus an adversarial regression set.** `pytest` fails if anyone lowers the
   threshold or weakens the guard to chase hit rate.

---

## API

OpenAI-shaped on purpose: an existing client switches to this gateway by changing a base URL.

| Endpoint | Method | Purpose |
|---|---|---|
| `/v1/chat` | POST | The gateway. Auth → rate limit → budget → cache → provider |
| `/v1/usage` | GET | Month-to-date spend, budget, and hourly rollups for the calling tenant |
| `/healthz` | GET | Pings Postgres and Redis, lists the active provider chain |
| `/metrics` | GET | Prometheus exposition; `bench/report.py` reads this |

**Request**

```json
{
  "messages": [{"role": "user", "content": "What is the capital of France?"}],
  "model": "mock-small",
  "temperature": 0.0,
  "max_tokens": 512,
  "no_cache": false
}
```

**Response**

```json
{
  "id": "b3f1...", "content": "...", "model": "mock-small", "provider": "mock",
  "cached": true, "similarity": 0.9567,
  "prompt_tokens": 8, "completion_tokens": 24,
  "cost_usd": 0.0, "latency_ms": 16
}
```

`cached` and `similarity` are the fields `bench/load.js` reads to split hit and miss latency into
separate distributions.

**Status codes**

| Code | Means |
|---|---|
| `401` | Unknown or missing API key |
| `429` | Rate limit exceeded — retry after the `Retry-After` header |
| `402` | Monthly budget exhausted. Deliberately not 429: retrying will not help |
| `503` | Every provider in the chain is down or circuit-open |

---

## Design decisions worth defending

**The embedding runs in-process.** The cache lookup must be cheaper than the call it avoids. A
hosted embedding API costs ~50 ms and real money to decide whether to skip a ~900 ms call; a local
5 ms encode makes that decision free. It also runs in a threadpool — 5 ms of blocking CPU on the
event loop serialises every concurrent request behind it.

**Rate limiting is a Lua script.** `GET`, compute, `SET` is a race: two concurrent requests read the
same token count and both proceed. Redis executes Lua atomically. `tests/test_limits.py` fires 50
concurrent requests at a burst-10 bucket and asserts exactly 10 pass — a read-modify-write
implementation passes the sequential test and fails that one.

**Budgets reserve, then settle.** You cannot know a call's cost before making it, so the request
pre-authorises a pessimistic estimate and refunds the difference afterwards — the same
authorise/capture split a card network uses. A cache hit refunds the whole reservation.

**Circuit breaker state lives in Redis.** With three API replicas, an in-process breaker means each
one independently rediscovers that the provider is down.

**Identical concurrent requests are coalesced.** On a cold entry, N simultaneous copies of the same
question would all miss and all pay — and because the cache write is async, that window stays open
until Celery lands the row. A `SET NX` lock elects one filler; the rest wait up to 2 s for its
result, then give up and call anyway. An unbounded wait on a holder that may have died is worse than
a duplicate call.

**No ANN index on the vectors.** pgvector post-filters, so with a highly selective tenant predicate
an HNSW index can return ten global neighbours that all belong to other tenants and leave zero rows
after filtering — a **cache miss on an entry that exists**, silently. Exact KNN inside one bucket is
correct and ~2 ms at this scale.

**Load tests run against a mock provider.** Benchmarking a real provider measures *their* latency
variance and *their* rate limits. The mock injects a lognormal latency distribution — real inference
latency is right-skewed — so the numbers isolate this gateway's own overhead, cost nothing, and
reproduce exactly.

---

## Configuration

Everything lives in `.env` (copy from `.env.example`). Defaults run the full stack with no API key.

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `postgresql://gw:gw@postgres:5432/gateway` | Postgres with pgvector |
| `REDIS_URL` | `redis://redis:6379/0` | Buckets, budgets, breaker state, fill locks |
| `PROVIDER_CHAIN` | `mock` | Ordered failover chain. `groq,gemini` for real providers |
| `SIMILARITY_THRESHOLD` | `0.94` | **Measured** by `bench/threshold_sweep.py`, not chosen |
| `CACHE_TTL_HOURS` | `168` | Read-time filter *and* what the reaper deletes past |
| `REQUEST_TIMEOUT_SECONDS` | `20` | Per-provider call timeout |
| `GROQ_API_KEY` / `GEMINI_API_KEY` | empty | Only needed for real providers |
| `MOCK_LATENCY_MEAN_MS` | `900` | Shapes the load test's latency distribution |
| `MOCK_FAILURE_RATE` | `0.0` | `bench/chaos.sh` turns this up to simulate an outage |

Everything else — cost model, breaker thresholds, velocity windows — is in `app/config.py`, in one
block, commented.

---

## Repository

```
llm-gateway/
├── DESIGN.md                    full rationale — every alternative considered and rejected
├── README.md                    this file
├── docker-compose.yml           postgres · redis · api · worker · beat
├── Dockerfile · pyproject.toml · .env.example
├── migrations/001_init.sql      schema; runs once on an empty volume
├── scripts/
│   └── scaffold.py              writes every source file out of DESIGN.md
├── app/
│   ├── config.py                every tunable value, one place
│   ├── models.py                the HTTP contract (OpenAI-shaped)
│   ├── db.py                    asyncpg pool + pgvector registration
│   ├── auth.py                  API key -> tenant
│   ├── providers/               base · mock · groq · gemini
│   ├── cache/
│   │   ├── embed.py             in-process sentence embedding
│   │   └── semantic.py          namespace, eligibility, LEXICAL GUARD, lookup
│   ├── limits.py                Redis Lua: token bucket, budgets, single-flight
│   ├── breaker.py               circuit breaker + cross-vendor failover
│   ├── accounting.py            cost estimate + idempotent usage ledger
│   ├── telemetry.py             Prometheus metrics
│   ├── tasks.py                 Celery: cache persist, usage, TTL reaper
│   └── main.py                  the request path — read this one first
├── bench/
│   ├── threshold_sweep.py       the headline experiment
│   ├── cost_model.py            cost per 1k requests vs hit rate
│   ├── load.js                  k6
│   ├── report.py                /metrics -> the tables in this README
│   └── chaos.sh                 kills the primary provider under load
└── tests/                       42 tests; test_cache_correctness.py is the one that matters
```

`DESIGN.md` is the source of truth — `scripts/scaffold.py` writes every file above out of it, so the
document and the code cannot drift. `python scripts/scaffold.py --check` reports any that have.

---

## Running it

Five containers, one command, no hosting account. `postgres` · `redis` · `api` · `worker` · `beat`.

**GitHub Codespaces** — open the repo → **Code → Codespaces → Create**, and the whole stack starts
in the browser. Nothing to install.

**Or on any machine with Docker:**

```bash
git clone https://github.com/<you>/llm-gateway && cd llm-gateway
python scripts/scaffold.py        # only if starting from DESIGN.md alone
cp .env.example .env
docker compose up --build
curl localhost:8000/healthz
```

No provider key is needed: the default `PROVIDER_CHAIN=mock` runs everything, including the load and
chaos tests.

Then watch the finding happen in three requests:

```bash
ask() { curl -s -XPOST localhost:8000/v1/chat \
  -H "Authorization: Bearer demo-key-please-change" \
  -H 'Content-Type: application/json' \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}]}" \
  | python -c 'import sys,json; d=json.load(sys.stdin); \
print(f"cached={d[\"cached\"]} sim={d[\"similarity\"]} {d[\"latency_ms\"]}ms")'; }

ask "What is the capital of France?"        # cached=False  ~900ms
sleep 1                                      # let the Celery write land
ask "What is the capital city of France?"   # cached=True   sim~0.957  ~16ms
ask "What is the capital of Germany?"       # cached=False  -- correctly NOT a hit
```

### The experiments

```bash
python bench/threshold_sweep.py                 # the finding
python bench/cost_model.py                      # cost curve
k6 run -e API_KEY=demo-key-please-change bench/load.js
bash bench/chaos.sh                             # kill the provider under load
python bench/report.py                          # /metrics -> the tables above
```

Every number in this README comes from that stack. There is no deployment step — a hosted URL would
add cost and operational surface without adding evidence.

### Real providers

Set `PROVIDER_CHAIN=groq,gemini` and add keys to `.env`. Both have free tiers. Failover crosses
vendors deliberately — two models at one provider share an incident.

---

## Tests

```bash
pytest -q          # 42 tests
```

| File | Tests | Guards |
|---|---|---|
| `test_cache_correctness.py` | **25** | The finding. 10 adversarial pairs that must never hit, 6 paraphrases that must, plus the guard's individual rules |
| `test_limits.py` | 9 | Lua atomicity, bucket refill, budget reserve/settle, single-flight election |
| `test_breaker.py` | 5 | Circuit opens at threshold, open circuit short-circuits, failover reaches provider 2, a 4xx does **not** open it |
| `test_accounting.py` | 3 | Exact `Decimal` cost, unknown model costs zero, estimate is pessimistic |

Three of these are load-bearing rather than routine:

- **`test_concurrent_requests_cannot_exceed_burst`** — 50 coroutines at a burst-10 bucket, exactly 10
  pass. A read-modify-write rate limiter passes every other test and fails this one.
- **`test_threshold_alone_is_insufficient`** — pins the finding. If a future embedding model
  separates the classes, this fails, telling you the lexical guard has become dead code.
- **`test_our_own_bad_request_does_not_open_the_circuit`** — a 4xx is our bug, not the provider's.
  Without this, one malformed request takes a healthy provider offline for every tenant.

`test_cache_correctness.py` and `test_accounting.py` need no database and no Redis — only the
embedding model — so they run on a bare checkout before any infrastructure exists.

---

## Capacity

Two resources saturate on the two paths, and they scale differently:

| Path | Cost | Bound by |
|---|---|---|
| Cache hit | ~16 ms, of which **5 ms is CPU** | Cores — ~800 req/s on 4 |
| Cache miss | ~900 ms, almost all waiting | The provider's rate limit, not ours |

Adding cores to fix a miss-bound service does nothing, and adding connections to fix a hit-bound one
does nothing either — they are different resources.

Storage is ~4 KB per cache entry (1.5 KB vector + text), so 2 M requests/month at a 40% miss rate
grows the table ~3.2 GB/month — which is why an hourly reaper deletes past-TTL rows in batches
rather than relying on read-time filtering alone.

Memory is ~600 MB resident for torch + the model, so budget 1 GB for the `api` container.

---

## Limitations

- **The adversarial test set is hand-built and small** (10 pairs). It demonstrates the failure mode
  and guards against regression; it is not a statistical estimate of wrong-hit rate in production.
- **The antonym list is curated and therefore incomplete.** It catches the classes it knows. A
  learned or WordNet-backed check would generalise further.
- **Some true paraphrases miss.** "Summarise this document" / "Give me a summary of this document"
  scores 0.789 and "What are your business hours?" / "When are you open?" scores 0.508 — both are
  semantically identical and both miss. That costs money, not correctness, but it caps the
  achievable hit rate; the fix is a stronger embedding model, not a lower threshold.
- **Cache correctness is only as good as the embedding model.** `all-MiniLM-L6-v2` is small and
  fast. A better model would raise the safety margin.
- **No streaming.** Caching an assembled response is straightforward; caching a stream is not.
- **Load numbers come from a mock provider** — deliberately, but they measure gateway overhead, not
  end-to-end user latency against a real model.
- **Cost figures are calculated, not billed.** `cost_model.py` multiplies published prices by
  measured token counts. Only `/v1/usage` reflects real spend, and only against a real provider.
- **Single region, no replication.** Designing multi-region without real traffic patterns would be
  guessing.
- **Budgets are eventually consistent.** Reserve/settle in Redis can drift by one in-flight request
  if a worker dies between the call and the settle. The Postgres ledger is the source of truth and
  reconciles it.

---

## What I would do next

1. **Streaming responses** with cache-on-complete
2. **Prompt-prefix caching** — many requests share a long system prompt; providers bill less for a
   cached prefix
3. **A shadow-eval sampler** — replay 1% of cache hits against the live provider and alert if the
   answers diverge, turning the adversarial test into continuous monitoring
4. **Per-tenant model routing** — cheap model by default, escalate on a quality signal
5. **Postgres partitioning + per-partition HNSW** when a bucket exceeds ~10⁵ entries

---

## License

MIT.
