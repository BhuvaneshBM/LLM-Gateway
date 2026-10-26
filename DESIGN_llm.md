# Sentry — LLM Inference Gateway · Design Document & Complete Source

**A production gateway between your applications and LLM providers: semantic caching, per-tenant
budgets, rate limiting, circuit breaking and cost accounting.**

This file is self-contained. Part I explains the design, **Part II contains every file in the
project in paste order**, Part III is the runbook and the experiments, Part IV is the interview
answer sheet.

---

## Contents

**Part I — Design**
1. [What this project is](#1-what-this-project-is)
2. [Why an LLM is a hard dependency](#2-why-an-llm-is-a-hard-dependency)
3. [The correctness problem nobody expects](#3-the-correctness-problem-nobody-expects)
4. [Architecture](#4-architecture)
5. [The four decisions that define the project](#5-the-four-decisions-that-define-the-project)

**Part II — Complete source**
6. [Folders to create](#6-folders-to-create) + [`scripts/scaffold.py`](#scriptsscaffoldpy--write-every-file-out-of-this-document)
7. [`pyproject.toml`](#7-pyprojecttoml) · [`.env.example`](#8-envexample) · [`.gitignore`](#9-gitignore)
8. [`migrations/001_init.sql`](#10-migrations001_initsql)
9. [`app/config.py`](#11-appconfigpy) · [`app/models.py`](#12-appmodelspy) · [`app/db.py`](#13-appdbpy)
10. [`app/providers/`](#14-appprovidersbasepy) — base · mock · groq · gemini
11. [`app/cache/`](#18-appcacheembedpy) — embed · semantic
12. [`app/limits.py`](#20-applimitspy) · [`app/breaker.py`](#21-appbreakerpy) · [`app/accounting.py`](#22-appaccountingpy)
13. [`app/telemetry.py`](#23-apptelemetrypy) · [`app/auth.py`](#24-appauthpy) · [`app/tasks.py`](#25-apptaskspy)
14. [`app/main.py`](#26-appmainpy) — the request path
15. [`tests/`](#27-tests) — including the adversarial cache set
16. [`bench/`](#28-bench) — load, threshold sweep, cost model, report, chaos
17. [Infrastructure](#29-dockerfile) — Dockerfile · compose · devcontainer
18. [`README.md`](#32-readmemd)

**Part III — Run it**
19. [Runbook](#33-runbook)
20. [What each file does, and the order to do it](#34-what-each-file-does-and-the-order-to-do-it) — [dependency map](#361-the-dependency-map) · [sequence](#362-the-sequence-start-to-end)
21. [The four experiments](#35-the-four-experiments)
22. [Failure modes](#36-failure-modes)

**Part IV — Interview prep**
23. [Why these choices and not the alternatives](#37-why-these-choices-and-not-the-alternatives)

---
---

# PART I — DESIGN

## 1. What this project is

A service that sits between your applications and LLM providers. Applications call it instead of
calling OpenAI/Groq/Gemini directly, and it provides the things every company adding LLM features
ends up needing:

| Capability | The problem it solves |
|---|---|
| **Semantic cache** | The same question asked slightly differently costs full price every time |
| **Per-tenant budgets** | One team's runaway loop burns the whole month's spend |
| **Rate limiting** | You get 429'd by the provider because five services share one key |
| **Circuit breaker + failover** | The provider has an incident and every feature in your product dies with it |
| **Cost accounting** | Nobody can answer "which team spent what" |

**Scope ends at a measured, documented service that runs locally.** No UI, no model training, no
fine-tuning, no hosting. Those are different projects with different skills.

**What a reviewer should conclude:** this person can reason about a slow, expensive,
non-deterministic dependency inside a latency budget; knows that caching has a correctness cost and
measured it; and thinks about infrastructure spend as a first-class design constraint.

### What was cut, and why

Scope discipline is itself a signal, so the removals are listed rather than hidden:

| Cut | Reason |
|---|---|
| **Streaming responses** | Real feature, but it complicates caching (you cache the assembled result) without adding a *new* design problem. Named as the first extension |
| **A UI / admin dashboard** | Prometheus + Grafana already renders everything worth seeing. Building a dashboard would be frontend work wearing a backend costume |
| **Fine-tuning / model hosting** | A different project. This treats the model as a dependency, deliberately |
| **Kubernetes** | `docker compose` runs the whole stack. k8s here would be résumé-driven architecture for five containers |
| **Multi-region** | Interesting, unmeasurable from one machine, and the honest answer is "I would need real traffic to design this" |
| **CI (GitHub Actions)** | A red badge on a portfolio repo is worse than no badge, and for a solo project CI catches nothing `pytest` does not. The workflow is four lines of YAML if you ever want it back — see the note in §33 |
| **Prometheus + Grafana** | The app exposes `/metrics` and `bench/report.py` scrapes it directly. Running a Prometheus container to read four histograms would be a service that exists to justify a bullet point |
| **Hosting it anywhere** | Every measurement in this project — latency percentiles, cache hit rate, cost curve, failover behaviour — comes from the local stack. A hosted URL costs money and adds operational surface without adding a single piece of evidence. `docker compose up` reproduces all of it |

Each is a reasonable next step; none is missing by accident.

---

## 2. Why an LLM is a hard dependency

This is the framing for the whole project. Strip the model out and you have a boring cache in front
of a fast API. Every hard problem here comes from six properties of the dependency:

| Property | Typical value | The design problem it forces |
|---|---|---|
| **Slow** | 400 ms – 3 s | A latency budget. What runs in parallel, what runs after the response is sent |
| **Expensive, per token** | \$0.05 – \$15 / M tokens | Cost modelling, cache ROI, per-tenant budgets |
| **Non-deterministic** | same prompt → different text | Which requests are cacheable *at all* |
| **Semantically addressed** | "capital of France" ≈ "France's capital city" | Vector search rather than a hash lookup — and see §3 |
| **Rate limited** | 429 under burst | Backpressure. Queue or shed, and for whom |
| **Occasionally down** | provider incidents | Circuit breaker, failover, degraded mode |

A hash-keyed cache in front of a fast, cheap, deterministic service is a first-week exercise. Every
one of those six properties is what makes this not that.

**The economics are worth internalising**, because they justify the entire project. At \$0.60 per
1,000 requests and 2 M requests/month, you are spending \$1,200/month. A 40 % cache hit rate saves
\$480/month. The gateway pays for itself at any realistic scale, and *that* is the sentence to lead
with when an interviewer asks why you built it.

---

## 3. The correctness problem nobody expects

Everything interesting in this project descends from one fact, and the specific shape of it is
**not** what most people assume.

> **A semantic cache can return a confidently wrong answer — and no similarity threshold can stop it.**

I measured this rather than assuming it, using `all-MiniLM-L6-v2`, and the result reversed my
starting hypothesis. All numbers below are cosine similarity, reproducible with
`python bench/threshold_sweep.py`.

**What embeddings handle fine.** The obvious worry — swapping an entity — is not the problem:

| Pair | Similarity | Verdict |
|---|---|---|
| "capital of **France**" / "capital of **Germany**" | **0.663** | safely far apart |
| "**start** the payment service" / "**stop** the payment service" | **0.725** | safely apart |
| "10 **largest** accounts" / "10 **smallest** accounts" | **0.824** | safely apart |
| "revenue in **Q1**" / "revenue in **Q3**" | **0.918** | apart |

**What breaks them.** Two classes, both far above any usable threshold:

| Pair | Similarity | Failure class |
|---|---|---|
| "Is this transaction **safe**?" / "Is this transaction **unsafe**?" | **0.965** | **negation** |
| "Convert 100 **USD to EUR**" / "Convert 100 **EUR to USD**" | **0.980** | **argument transposition** |
| "Transfer funds **savings → checking**" / "**checking → savings**" | **0.992** | **argument transposition** |

Now put those next to legitimate paraphrases that a cache *must* catch to be worth having:

| Pair | Similarity |
|---|---|
| "What is the capital of France?" / "What's the capital city of France?" | 0.957 |
| "List all active users" / "List every active user" | 0.956 |
| "How do I reset my password?" / "How can I reset my password?" | 0.992 |

### The finding

**0.992 is a wrong answer. 0.956 is a right one.** The classes overlap, so *there is no threshold
that admits real paraphrases and rejects argument transposition.* Tuning cannot fix this. Raising
the threshold to 0.995 to exclude the funds-transfer bug also excludes "capital city of France" and
"every active user" — you get a safe cache that never fires.

This is the project's actual result, and it is worth more than the working code: **a naive semantic
cache is unsafe at every operating point, and the standard advice to "tune the threshold" is wrong.**

### The fix: cosine for recall, lexical check for precision

Embeddings measure *topical* similarity. Negation and argument order are *structural* properties
that survive embedding almost untouched — "not" contributes little to a sentence vector, and
bag-of-words models are order-insensitive by construction. So the guard has to look at the thing the
embedding discards:

| Rule | Catches | How |
|---|---|---|
| **Identical token multiset, different order** | argument transposition | `sorted(tokens(a)) == sorted(tokens(b)) and tokens(a) != tokens(b)` |
| **Polarity token on exactly one side** | explicit negation | `{"not","no","never","without",…} ∩ (A △ B)` |
| **Morphological negation** | `safe` / `unsafe` | a differing token whose `un`/`non`/`in`/`dis`-stripped stem appears on the other side |
| **Known antonym pair split across sides** | `read`/`write`, `grant`/`revoke` | small curated list |

Measured over 10 adversarial and 6 paraphrase pairs at threshold 0.94:

| | Cosine alone | Cosine + lexical guard |
|---|---|---|
| Wrong hits | **2 / 10** | **0 / 10** |
| Useful hits | 6 / 6 | **6 / 6** |

The guard costs about 20 µs of string work — three orders of magnitude below the network call it is
protecting — and it removes every wrong hit without giving up a single legitimate one.

### The four guards, in order of how much they buy you

**1. Namespacing (structural).** Similarity is only ever consulted *within* a bucket keyed by
`(tenant, model, system_prompt_hash, temperature_bucket)`. A cross-tenant or cross-model hit is
impossible by construction. Most of the safety lives here.

**2. Eligibility (structural).** Only `temperature ≤ 0.3`, no tool calls, bounded length. A request
asking for creative variation is not cacheable, and pretending otherwise is the bug.

**3. The lexical guard.** Above. This is what makes the cache safe *at a usable threshold*.

**4. A measured threshold plus an adversarial test set.** The threshold comes from the sweep; the
test set fails if anyone lowers it or weakens the guard to chase hit rate.

That last one is this project's equivalent of a leakage test in an ML pipeline: a cheap check
guarding the one bug that would otherwise be invisible and expensive.

### The honest limitation

The antonym list is curated and therefore incomplete — it catches the classes it knows. Two
paraphrase pairs I originally wrote also turned out to be *false negatives* for a different reason:
"Summarise this document" / "Give me a summary of this document" scores **0.789**, and "What are your
business hours?" / "When are you open?" scores **0.508**. Both are semantically identical and both
miss the cache. That costs money, not correctness — a miss is safe — but it caps the achievable hit
rate, and the fix is a stronger embedding model rather than a lower threshold.

---

## 4. Architecture

```
                     ┌──────────── SYNC PATH — p95 budget 1.5 s ─────────────┐
   POST /v1/chat     │                                                        │
   Authorization: …  │  FastAPI                                               │
   ─────────────────►│    1. auth          tenant from API key      ~1 ms     │
                     │    2. rate limit    Redis Lua, token bucket  ~1 ms     │
                     │    3. budget        Redis Lua, spend cap     ~1 ms     │
                     │    4. eligible?     temp/tools/length         ~0 ms    │
                     │    5. embed         sentence-transformers    ~5 ms     │
                     │    6. cache lookup  pgvector, namespaced     ~8 ms     │
                     │         │                                              │
                     │         ├── HIT  ──────────────► return   ~16 ms $0    │
                     │         │                                              │
                     │         └── MISS ─► breaker ─► provider   ~900 ms $$   │
                     │                        │  primary → fallback           │
                     └────────────────────────┼──────────────────────────────┘
                                              │ enqueue, do not await
                     ┌────────────────────────▼──────────────────────────────┐
                     │  Celery                                               │
                     │    persist_cache_entry   embedding + response         │
                     │    record_usage          tokens, cost, tenant         │
                     │    rollup_hourly         per-tenant spend             │
                     └───────────────────────────────────────────────────────┘

   Redis  token buckets · spend counters · breaker state
   Postgres + pgvector  cache entries · tenants · usage ledger
```

**The sync/async split is the design.** Everything the caller must wait for is on the top path and
costs ~16 ms before the provider call. Everything else — writing the cache entry, embedding storage,
cost accounting, rollups — happens after the response is already on its way back.

That ordering is deliberate and worth defending: **the cache write is the single most tempting thing
to do synchronously** and the single worst. It adds a Postgres write to every miss, on the path the
user is waiting on, to benefit *some future request*. Moving it to Celery costs nothing except that
two identical requests arriving within the same few hundred milliseconds both miss. That is a trade
worth making, and being able to say so is the point.

---

### 4.1 Capacity and bottlenecks

Two different resources saturate on the two paths, and they scale differently. Knowing which is
which is most of what a capacity question is asking.

| Path | Cost | Bound by |
|---|---|---|
| Embedding | **5 ms CPU** | Cores. 4 cores ≈ **800 req/s ceiling** |
| Cache lookup | 8 ms I/O | Postgres connections, not CPU |
| **Cache hit, total** | **~16 ms** | CPU (the embedding) |
| **Cache miss, total** | **~900 ms**, almost all waiting | The *provider's* rate limit, not ours |

**The embedding is the CPU bottleneck; the provider is the latency bottleneck.** A cache hit burns a
core for 5 ms and returns; a cache miss burns almost no CPU and holds a coroutine for a second. So
hit throughput scales with cores and miss concurrency scales with connections and memory — adding
cores to fix a miss-bound service does nothing.

**Storage.** 384 floats × 4 bytes = 1.5 KB per vector, plus ~2.5 KB of prompt and response text:

```
  ~4 KB per cache entry
  2M requests/month x 40% miss  = 800K new entries/month
                                ≈ 3.2 GB/month
```

That number is why the reaper in §25 exists. TTL filtering at read time hides expired rows but does
not delete them, so without a reaper the table and its index grow forever and the exact-KNN scan
degrades — slowly, invisibly, and in production.

**Memory.** torch plus `all-MiniLM-L6-v2` resident is ~600 MB, so budget **1 GB for the api
container**. Below about 700 MB it OOMs *during model load*, and the symptom is a restart loop
rather than an out-of-memory message — worth knowing if you ever constrain the container.

> If you later need this to fit in a 512 MB box, swap `sentence-transformers` for **`fastembed`**,
> which runs the same `all-MiniLM-L6-v2` through ONNX Runtime: ~150 MB resident, ~300 ms to load
> instead of ~2 s, and faster CPU inference. Same vectors, so §3's numbers are unchanged.

**Where it breaks at 10×.** Two different failures needing two different answers:

- *Hit-heavy traffic* saturates CPU on embedding. The service is stateless — all shared state is in
  Redis and Postgres — so this is horizontal scaling and nothing else.
- *Miss-heavy traffic* saturates the provider's rate limit, which no amount of scaling fixes. That
  is where the real decision lives: **queue or shed**. Shed for free tenants (a 429 is more honest
  than a 30-second wait), queue for paying ones.

---

## 5. The four decisions that define the project

**(a) The embedding runs in-process, not over the network.**
The cache lookup must be cheaper than the call it is avoiding. A hosted embedding API costs ~50 ms
and real money to decide whether to skip a ~900 ms call — the arithmetic still works, but a 5 ms
local encode makes the decision free. `all-MiniLM-L6-v2` is 80 MB, 384 dimensions, and runs on CPU.
This is the highest ratio of "obvious in hindsight" to "commonly got wrong" in the project.

**(b) Rate limiting is a Lua script, not read-then-write.**
`GET` the bucket, compute, `SET` it back is a race. Two concurrent requests from the same tenant both
read the same token count and both proceed. Redis executes a Lua script atomically, so the
check-and-decrement cannot interleave. This is the specific detail that separates "I know what a
token bucket is" from "I have implemented one that works under concurrency."

**(c) All load testing runs against a mock provider.**
Benchmarking against a real provider measures *their* latency variance and *their* rate limits, not
your gateway. `MockProvider` injects a configurable latency distribution and failure rate, so the
numbers isolate your own overhead, cost nothing, and reproduce exactly. This is a methodology
decision, not a shortcut, and it belongs in the README.

**(d) The cache is guarded lexically, not only by threshold.**
Measurement showed the adversarial and paraphrase similarity distributions **overlap** (§3): an
argument transposition scores 0.992 while a legitimate paraphrase scores 0.956, so no threshold
separates them. Cosine provides recall; a ~20 µs lexical check on token order, polarity and antonyms
provides precision. Picking 0.9 because it looks like a good number is the failure this project
exists to demonstrate you would not commit — but the deeper point is that *no* number would have
worked.

---
---

# PART II — COMPLETE SOURCE

Create each file at the exact path in its heading. Order matters only in that later files import
earlier ones.

## 6. Folders to create

```
llm-gateway/
├── .devcontainer/
├── app/{providers,cache}/
├── bench/
├── migrations/
└── tests/
```

```bash
mkdir -p llm-gateway/{.devcontainer,app/providers,app/cache,bench/results,migrations,tests}
cd llm-gateway && git init

# Package markers. `app` is imported as a package by every module, and
# `pip install -e .` (which the Dockerfile runs) needs them present.
touch app/__init__.py app/providers/__init__.py app/cache/__init__.py

# Keeps bench/results/ in git while .gitignore excludes the generated
# JSON and PNGs inside it.
touch bench/results/.gitkeep
```

All three `__init__.py` files are intentionally **empty**. There is no package-level
API worth exporting here, and re-exporting names in `__init__.py` makes the import
graph harder to follow than `from app.cache import semantic` already is.


### `scripts/scaffold.py` — write every file out of this document

Pasting 37 files by hand is an afternoon of tedium and the easiest way to introduce a typo that costs
an hour to find. Paste **this one file** next to `DESIGN.md` and run it instead.

```python
"""Extract every source file from DESIGN.md into the working tree.

    python scripts/scaffold.py            # write files
    python scripts/scaffold.py --check    # report differences, write nothing

DESIGN.md is the single source of truth: each file lives under a heading naming
its path, followed by a fenced block. This reads exactly that and nothing else,
so the document and the tree cannot drift.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "DESIGN.md"

# "## 12. `app/models.py`"  /  "### `tests/conftest.py` — the important one"
HEADING = re.compile(
    r"^#{2,4}\s+(?:\d+\.\s*)?`([^`]+)`(?:\s+[-—].*)?\s*$", re.M)

# Only these become files. Everything else in the doc is prose or illustration.
SUFFIXES = {".py", ".toml", ".sql", ".yml", ".yaml", ".json", ".js", ".sh",
            ".md", ".example", ".gitignore"}
NAMED = {"Dockerfile", ".gitignore", ".env.example"}

# Some sections show a usage example before the code. Matching the fence
# language to the file type takes the RIGHT block regardless of order -- this
# script's own section is the case that found the bug.
LANGS = {
    ".py": {"python", "py"}, ".toml": {"toml"}, ".sql": {"sql"},
    ".yml": {"yaml", "yml"}, ".yaml": {"yaml", "yml"}, ".json": {"json"},
    ".js": {"javascript", "js"}, ".sh": {"bash", "sh"},
    ".md": {"markdown", "md"}, ".example": {"bash", "sh"},
}
NAMED_LANGS = {"Dockerfile": {"dockerfile"}, ".gitignore": {"gitignore"},
               ".env.example": {"bash", "sh"}}

# Written after extraction; they have no content worth putting in the document.
EMPTY_FILES = [
    "app/__init__.py",
    "app/providers/__init__.py",
    "app/cache/__init__.py",
    "bench/results/.gitkeep",
]


def is_file_heading(name: str) -> bool:
    return name in NAMED or Path(name).suffix in SUFFIXES


def extract(text: str) -> list[tuple[str, str]]:
    """Return [(path, body)] for the best fenced block under each file heading.

    "Best" = the first block whose fence language matches the file type, falling
    back to the first block if none is tagged. A section that shows a `bash`
    usage example before its code would otherwise write the example into the
    source file.
    """
    lines = text.splitlines()
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(lines):
        m = HEADING.match(lines[i])
        if not m or not is_file_heading(m.group(1)):
            i += 1
            continue
        path = m.group(1)

        want = NAMED_LANGS.get(path) or LANGS.get(Path(path).suffix) or set()
        blocks: list[tuple[str, str]] = []          # (language, body)
        j = i + 1
        while j < len(lines):
            if HEADING.match(lines[j]):             # next file section
                break
            if not lines[j].startswith("```"):
                j += 1
                continue
            # README.md is fenced with four backticks because it contains three.
            fence = "````" if lines[j].startswith("````") else "```"
            lang = lines[j][len(fence):].strip().lower()
            k = j + 1
            body: list[str] = []
            while k < len(lines) and not lines[k].startswith(fence):
                body.append(lines[k])
                k += 1
            blocks.append((lang, chr(10).join(body).rstrip(chr(10)) + chr(10)))
            j = k + 1
            if lang in want:                        # right one, stop looking
                break

        if blocks:
            out.append((path, next((b for lang, b in blocks if lang in want),
                                   blocks[0][1])))
        i = j
    return out


def verify(path: str, body: str) -> str | None:
    """Cheap syntax check so a bad paste fails here rather than at runtime."""
    try:
        if path.endswith(".py"):
            ast.parse(body, filename=path)
        elif path.endswith(".json"):
            json.loads(body)
    except (SyntaxError, ValueError) as e:
        return f"{type(e).__name__}: {e}"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report differences without writing")
    args = ap.parse_args()

    if not DOC.exists():
        print(f"cannot find {DOC}", file=sys.stderr)
        return 1

    files = extract(DOC.read_text(encoding="utf-8"))
    if not files:
        print("no file blocks found -- has the heading format changed?", file=sys.stderr)
        return 1

    written = changed = same = 0
    problems: list[str] = []

    for path, body in files:
        err = verify(path, body)
        if err:
            problems.append(f"{path}: {err}")
            continue

        dest = ROOT / path
        existing = dest.read_text(encoding="utf-8") if dest.exists() else None

        if existing == body:
            same += 1
            continue
        if args.check:
            print(f"  DIFFERS  {path}" if existing is not None else f"  NEW      {path}")
            changed += 1
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(body, encoding="utf-8")
        print(f"  {'updated' if existing is not None else 'wrote  '}  {path}")
        written += 1

    if not args.check:
        for rel in EMPTY_FILES:
            p = ROOT / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            if not p.exists():
                p.touch()
                print(f"  wrote    {rel}")

    print(f"\n{len(files)} file blocks | {written} written | {same} unchanged"
          + (f" | {changed} would change" if args.check else ""))

    if problems:
        print("\nPROBLEMS -- these were NOT written:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        return 1

    if not args.check:
        print("\nnext:  cp .env.example .env && docker compose up --build")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> **Does** — writes every source file in this document out to disk, so `DESIGN.md` stays the single
> source of truth and the tree cannot drift from it.
> **Watch out** — it matches the **fence language** to the file type rather than taking the first
> block, because several sections show a `bash` usage example before the code. `--check` reports
> drift without writing; run it before committing and the doc and the code can never disagree.

`--check` is the useful mode once you are working: it tells you whether the tree has drifted from the
document without touching anything. Run it before committing and you will never ship a repo whose
code and design doc disagree.

---

## 7. `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-gateway"
version = "0.1.0"
description = "LLM inference gateway: semantic cache, budgets, rate limiting, failover"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.27",
    "httpx>=0.27",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "asyncpg>=0.29",
    "pgvector>=0.2.5",
    "redis>=5.0",
    "celery>=5.3",
    "sentence-transformers>=2.5",
    "prometheus-client>=0.20",
    "structlog>=24.1",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.3",
    "matplotlib>=3.8",     # bench/ plots only
    "numpy>=1.26",
]

# WITHOUT this, `pip install -e .` fails with "Multiple top-level packages
# discovered in a flat-layout: app, bench, tests, migrations". setuptools
# auto-discovery refuses to guess, and the Dockerfile build dies at that line.
# `bench`, `tests` and `migrations` are scripts and fixtures, not library code,
# so they are deliberately not installed.
[tool.setuptools]
packages = ["app", "app.providers", "app.cache"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"
```

> **Does** — declares dependencies and makes `app` an installable package.
> **Watch out** — `[tool.setuptools] packages` is not optional. Without it `pip install -e .` fails
> with *"Multiple top-level packages discovered in a flat-layout"* and the Docker build dies on that
> line. `bench`, `tests` and `migrations` are scripts and fixtures, so they are excluded on purpose.

---

## 8. `.env.example`

```bash
# Copy to .env. docker-compose reads it; NEVER commit the real one.
DATABASE_URL=postgresql://gw:gw@postgres:5432/gateway
REDIS_URL=redis://redis:6379/0

# "mock" needs no key and is what every load test uses (§5c).
# Add real providers only for the demo run.
PROVIDER_CHAIN=mock
GROQ_API_KEY=
GEMINI_API_KEY=

# The number chosen by bench/threshold_sweep.py -- do not guess it (§5d).
SIMILARITY_THRESHOLD=0.94
```

> **Does** — the local configuration template; `docker compose` reads the copy you make at `.env`.
> **Watch out** — `PROVIDER_CHAIN=mock` is what lets someone clone this and run the entire stack,
> including every load and chaos test, **without an API key**. Keep it as the default.

---

## 9. `.gitignore`

```gitignore
.env
__pycache__/
*.pyc
.venv/
.pytest_cache/
.ruff_cache/
bench/results/*.json
bench/results/*.png
!bench/results/.gitkeep
```

> **Does** — keeps secrets (`.env`) and generated artifacts out of git.
> **Watch out** — the `!bench/results/.gitkeep` negation only works if that file exists, which is why
> `scaffold.py` creates it. Force-add any figure the README links to, or you ship a broken image.

> Force-add the plots your README links to: `git add -f bench/results/threshold_curve.png`.
> A README pointing at images nobody can see is worse than no README.

---

## 10. `migrations/001_init.sql`

```sql
-- Run once at container start (docker-compose mounts this into postgres init).
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tenants (
    id                  BIGSERIAL PRIMARY KEY,
    name                TEXT        NOT NULL UNIQUE,
    api_key_hash        TEXT        NOT NULL UNIQUE,
    rate_per_sec        REAL        NOT NULL DEFAULT 10,
    burst               INTEGER     NOT NULL DEFAULT 20,
    monthly_budget_usd  NUMERIC(12,4) NOT NULL DEFAULT 50,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (rate_per_sec > 0 AND burst > 0 AND monthly_budget_usd >= 0)
);

-- One row per cached completion.
--
-- `namespace` is hash(model, system_prompt, temperature_bucket). Similarity is
-- ONLY ever compared within one (tenant_id, namespace) bucket, so a cross-tenant
-- or cross-model hit is structurally impossible rather than threshold-dependent.
CREATE TABLE IF NOT EXISTS cache_entries (
    id                BIGSERIAL PRIMARY KEY,
    tenant_id         BIGINT      NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    namespace         TEXT        NOT NULL,
    prompt            TEXT        NOT NULL,
    embedding         vector(384) NOT NULL,
    response          TEXT        NOT NULL,
    prompt_tokens     INTEGER     NOT NULL,
    completion_tokens INTEGER     NOT NULL,
    provider          TEXT        NOT NULL,
    model             TEXT        NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The lookup index. Deliberately NOT an HNSW index -- see the note below.
CREATE INDEX IF NOT EXISTS cache_entries_bucket
    ON cache_entries (tenant_id, namespace, created_at DESC);

-- Append-only usage ledger. request_id is UNIQUE so a Celery retry cannot
-- double-count spend -- the task is idempotent by database constraint rather
-- than by hoping it runs once.
CREATE TABLE IF NOT EXISTS usage_events (
    id                BIGSERIAL PRIMARY KEY,
    tenant_id         BIGINT      NOT NULL REFERENCES tenants(id),
    request_id        UUID        NOT NULL UNIQUE,
    cache_hit         BOOLEAN     NOT NULL,
    provider          TEXT,
    model             TEXT,
    prompt_tokens     INTEGER     NOT NULL DEFAULT 0,
    completion_tokens INTEGER     NOT NULL DEFAULT 0,
    cost_usd          NUMERIC(14,8) NOT NULL DEFAULT 0,
    latency_ms        INTEGER     NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (cost_usd >= 0)
);

CREATE INDEX IF NOT EXISTS usage_events_tenant_time
    ON usage_events (tenant_id, created_at DESC);

-- Seed a demo tenant. api_key_hash = sha256("demo-key-please-change").
INSERT INTO tenants (name, api_key_hash, rate_per_sec, burst, monthly_budget_usd)
VALUES ('demo',
        'c9f0f895fb98ab9159f51fd0297e236d6c3e1a1b7c8f0b3b6a2f5e7a1d4c8e2f',
        50, 100, 25.00)
ON CONFLICT (name) DO NOTHING;
```

> **Does** — the whole schema: `tenants`, `cache_entries`, `usage_events`. Runs once, automatically,
> when Postgres initialises an empty volume.
> **Two deliberate decisions live here** — there is **no ANN index** (see the note directly below,
> post-filtering would silently return zero rows), and `usage_events.request_id` is **UNIQUE**, which
> is what makes a Celery retry unable to double-count spend.

### Why there is no HNSW index

The obvious move is `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops)`. It is wrong here,
and knowing why is worth more than having the index.

pgvector's ANN indexes perform **post-filtering**: the index finds the globally nearest neighbours,
*then* the `WHERE tenant_id = … AND namespace = …` predicate is applied. With a highly selective
filter — and ours is extremely selective — the index can return ten global neighbours that all
belong to other tenants, leaving zero rows after filtering. **You get a cache miss on an entry that
exists**, silently and unpredictably.

Filtering on the btree first and running an exact KNN scan within one bucket is *correct*, and at a
few thousand rows per bucket it is also fast — a 384-dimensional cosine distance over 5,000 rows is
around 2 ms.

**What changes at scale:** past roughly 10⁵ entries per bucket, partition `cache_entries` by tenant
and build a per-partition HNSW index, so the ANN search runs inside an already-filtered set. That is
the answer to "how would this scale", and it is a better answer than having added the wrong index
early.

---

## 11. `app/config.py`

```python
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
```

> **Does** — every tunable value in the project: paths, column names, cache threshold and TTL,
> provider chain, prices, breaker settings, mock latency shaping.
> **Used by** — everything. It imports nothing from the project, so it is the bottom of the graph.
> **Watch out** — `SIMILARITY_THRESHOLD` is a **measured** number from `bench/threshold_sweep.py`, not
> a chosen one. If you edit it by hand, re-run `tests/test_cache_correctness.py` before trusting it.

---

## 12. `app/models.py`

```python
"""Request/response contracts. Deliberately OpenAI-shaped so existing clients
can point at this gateway by changing a base URL and nothing else."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    # Escape hatch: a caller that knows its request is unique skips the lookup
    # and saves ~13 ms. Also how the load test measures the uncached path.
    no_cache: bool = False


class ChatResponse(BaseModel):
    id: str
    content: str
    model: str
    provider: str
    cached: bool
    similarity: float | None = None
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    latency_ms: int
```

> **Does** — the HTTP contract. `Message`, `ChatRequest`, `ChatResponse`.
> **Watch out** — the shape is OpenAI-compatible on purpose, so an existing client switches to this
> gateway by changing a base URL and nothing else. `no_cache` is the escape hatch that lets a caller
> skip the ~13 ms lookup — and it is how `bench/load.js` measures the uncached path.

---

## 13. `app/db.py`

```python
"""asyncpg pool. One pool per process, created on startup."""

from __future__ import annotations

import asyncpg
from pgvector.asyncpg import register_vector

from .config import DATABASE_URL

_pool: asyncpg.Pool | None = None


async def _init(conn: asyncpg.Connection) -> None:
    # Teaches asyncpg the pgvector type, so a numpy array binds to vector(384)
    # directly instead of being stringified.
    await register_vector(conn)


async def connect() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            DATABASE_URL, min_size=2, max_size=10, init=_init,
            command_timeout=10,
        )
    return _pool


async def close() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("db.connect() has not run -- check the lifespan handler")
    return _pool
```

> **Does** — creates and holds the single asyncpg pool.
> **Exports** — `connect()` · `close()` · `pool()`.
> **Watch out** — `register_vector` runs in the pool's `init` hook, per connection. Without it a numpy
> array is stringified instead of binding to `vector(384)`, and the insert fails with a type error
> that points nowhere near the real cause.

---

## 14. `app/providers/base.py`

```python
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
```

> **Does** — the seam that makes everything downstream provider-agnostic. `Provider` protocol,
> `Usage`, `Completion`, `ProviderError`, and `cost_usd()`.
> **Watch out** — `ProviderError.retryable` carries real weight. A 4xx means *we* sent something
> malformed; counting it as a breaker failure would open the circuit on a healthy provider and turn
> our bug into an outage. Costs are `Decimal` because floats accumulate error over millions of rows
> and this number feeds a spend cap.

---

## 15. `app/providers/mock.py`

```python
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
```

> **Does** — a deterministic provider with a shaped latency distribution and a failure dial.
> **Used by** — **every load and chaos test.** This is a methodology decision, not a shortcut.
> **Watch out** — latency is **lognormal**, because real inference latency is right-skewed: a long
> tail of slow responses and never a negative one. A uniform distribution would make your p99 look
> far better than reality. `failure_rate` is what `chaos.sh` turns up to simulate an outage.

---

## 16. `app/providers/groq.py`

```python
"""Groq. OpenAI-compatible endpoint, generous free tier, very fast."""

from __future__ import annotations

import httpx

from ..config import GROQ_API_KEY, REQUEST_TIMEOUT_SECONDS
from .base import Completion, ProviderError, Usage

URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider:
    name = "groq"

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)

    async def complete(self, messages, model, temperature, max_tokens) -> Completion:
        try:
            r = await self._client.post(
                URL,
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": model, "messages": messages,
                      "temperature": temperature, "max_tokens": max_tokens},
            )
        except httpx.TimeoutException as e:
            raise ProviderError(self.name, f"timeout: {e}") from e
        except httpx.HTTPError as e:
            raise ProviderError(self.name, f"transport: {e}") from e

        # 429 and 5xx are retryable elsewhere; 4xx is our bug and must not trip
        # the breaker -- opening the circuit because we sent a malformed request
        # would take down a healthy provider.
        if r.status_code == 429 or r.status_code >= 500:
            raise ProviderError(self.name, f"http {r.status_code}", retryable=True)
        if r.status_code >= 400:
            raise ProviderError(self.name, f"http {r.status_code}: {r.text[:200]}",
                                retryable=False)

        body = r.json()
        usage = body.get("usage", {})
        return Completion(
            text=body["choices"][0]["message"]["content"],
            usage=Usage(prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0)),
            provider=self.name,
            model=body.get("model", model),
        )
```

> **Does** — the primary real provider. OpenAI-compatible endpoint, generous free tier, very fast.
> **Watch out** — the status-code split is the important part: **429 and 5xx are retryable** and trip
> the breaker; **4xx is not** and is raised immediately. Getting that backwards means one malformed
> request takes a healthy provider offline for everyone.

---

## 17. `app/providers/gemini.py`

```python
"""Gemini. Serves as the FAILOVER target -- a different vendor, so a Groq
incident and a Gemini incident are uncorrelated. Failing over to a second model
on the same provider would protect against almost nothing."""

from __future__ import annotations

import httpx

from ..config import GEMINI_API_KEY, REQUEST_TIMEOUT_SECONDS
from .base import Completion, ProviderError, Usage

BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class GeminiProvider:
    name = "gemini"

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS)

    async def complete(self, messages, model, temperature, max_tokens) -> Completion:
        # Gemini splits the system instruction out and calls the rest "contents",
        # with "model" where OpenAI says "assistant".
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            {"role": "model" if m["role"] == "assistant" else "user",
             "parts": [{"text": m["content"]}]}
            for m in messages if m["role"] != "system"
        ]
        payload = {
            "contents": contents,
            "generationConfig": {"temperature": temperature,
                                 "maxOutputTokens": max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}

        try:
            r = await self._client.post(
                f"{BASE}/{model}:generateContent",
                params={"key": GEMINI_API_KEY}, json=payload,
            )
        except httpx.TimeoutException as e:
            raise ProviderError(self.name, f"timeout: {e}") from e
        except httpx.HTTPError as e:
            raise ProviderError(self.name, f"transport: {e}") from e

        if r.status_code == 429 or r.status_code >= 500:
            raise ProviderError(self.name, f"http {r.status_code}", retryable=True)
        if r.status_code >= 400:
            raise ProviderError(self.name, f"http {r.status_code}: {r.text[:200]}",
                                retryable=False)

        body = r.json()
        try:
            text = body["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            # A 200 with no candidates means a safety block. That is a provider
            # failure from our side, not a transport error.
            raise ProviderError(self.name, f"no candidate: {str(body)[:200]}") from e

        meta = body.get("usageMetadata", {})
        return Completion(
            text=text,
            usage=Usage(prompt_tokens=meta.get("promptTokenCount", 0),
                        completion_tokens=meta.get("candidatesTokenCount", 0)),
            provider=self.name,
            model=model,
        )
```

> **Does** — the failover target. A **different vendor**, deliberately.
> **Watch out** — failing over to a second model at the same provider protects against almost
> nothing, since both share an incident. Gemini also splits the system instruction out of `contents`
> and says `model` where OpenAI says `assistant`, so the payload is reshaped here. A 200 with no
> candidates is a safety block, which is a provider failure from our side, not a transport error.

---

## 18. `app/cache/embed.py`

```python
"""In-process embedding.

Loaded once at startup, ~80 MB resident, ~5 ms per encode on CPU. The model is
loaded eagerly rather than lazily: a lazy load makes the FIRST request after
every restart pay a 2-3 s penalty, and it lands in your p99 rather than in
startup where it belongs.
"""

from __future__ import annotations

import numpy as np

from ..config import EMBED_DIM, EMBED_MODEL

_model = None


def load() -> None:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)


def encode(text: str) -> np.ndarray:
    """L2-normalised float32 vector.

    Normalising here means cosine distance and inner product agree, and pgvector's
    `<=>` operator returns a value we can convert to similarity with `1 - d`
    without worrying about magnitude.
    """
    if _model is None:
        load()
    v = _model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    v = np.asarray(v, dtype=np.float32)
    assert v.shape == (EMBED_DIM,), f"expected {EMBED_DIM}-d, got {v.shape}"
    return v
```

> **Does** — turns a prompt into a 384-dimension L2-normalised vector, in-process, in ~5 ms.
> **Exports** — `load()` · `encode()`.
> **Watch out** — `encode()` is **CPU-bound and blocking**. Calling it directly on the event loop
> stalls every other in-flight request for its duration; `main.py` must call it through
> `run_in_threadpool`. The model loads eagerly at startup so the 2–3 s cost lands in boot rather than
> in someone's p99.

---

## 19. `app/cache/semantic.py`

```python
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
```

> **Does** — the cache and three of its four guards. `namespace()` · `query_text()` ·
> `is_eligible()` · **`lexical_conflict()`** · `lookup()` · `store()`.
> **Watch out** — `lookup()` takes the **query text**, not only the vector. The lexical guard compares
> word order and polarity against the stored prompt, and it cannot do that from an embedding. If you
> ever refactor the signature and drop `query`, the guard silently stops working and the cache starts
> serving argument-transposed prompts again — the exact bug this project exists to demonstrate.

---

## 20. `app/limits.py`

```python
"""Rate limiting and budgets. Both are Lua scripts, and that is the point.

GET the bucket, compute in Python, SET it back is a RACE. Two concurrent
requests from the same tenant read the same token count, both decide they have
room, and both proceed -- so a 10 rps limit passes 20. Redis runs a Lua script
atomically against a single-threaded event loop, so the check and the decrement
cannot interleave.

This is the difference between knowing what a token bucket is and having built
one that survives concurrency.
"""

from __future__ import annotations

import time
from decimal import Decimal

import redis.asyncio as aioredis

from .config import REDIS_URL

_redis: aioredis.Redis | None = None

# ---------------------------------------------------------------- scripts --
_TOKEN_BUCKET = """
local rate  = tonumber(ARGV[1])
local burst = tonumber(ARGV[2])
local now   = tonumber(ARGV[3])
local cost  = tonumber(ARGV[4])

local state  = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts     = tonumber(state[2])

if tokens == nil then          -- first request for this tenant
  tokens = burst
  ts = now
end

local elapsed = now - ts
if elapsed < 0 then elapsed = 0 end
tokens = math.min(burst, tokens + elapsed * rate)

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
-- Expire after a full refill plus slack, so idle tenants do not leak keys.
redis.call('EXPIRE', KEYS[1], math.ceil(burst / rate) + 60)

return {allowed, tostring(tokens)}
"""

# Pre-authorise against an ESTIMATE, then settle with the actual cost once the
# provider replies -- the same authorise/capture split a card network uses. You
# cannot know the true cost before the call, and refusing to cap spend until
# after you have spent it is not a cap.
_RESERVE = """
local budget = tonumber(ARGV[1])
local amount = tonumber(ARGV[2])
local ttl    = tonumber(ARGV[3])

local spent = tonumber(redis.call('GET', KEYS[1]) or '0')
if spent >= budget then
  return {0, tostring(spent)}
end

local updated = spent + amount
redis.call('SET', KEYS[1], tostring(updated), 'EX', ttl)
return {1, tostring(updated)}
"""

_ADJUST = """
local delta = tonumber(ARGV[1])
local ttl   = tonumber(ARGV[2])
local spent = tonumber(redis.call('GET', KEYS[1]) or '0') + delta
if spent < 0 then spent = 0 end
redis.call('SET', KEYS[1], tostring(spent), 'EX', ttl)
return tostring(spent)
"""

_MONTH_TTL = 40 * 24 * 3600      # comfortably longer than any calendar month

_bucket_sha = _reserve_sha = _adjust_sha = None


async def connect() -> aioredis.Redis:
    global _redis, _bucket_sha, _reserve_sha, _adjust_sha
    if _redis is None:
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        _bucket_sha = await _redis.script_load(_TOKEN_BUCKET)
        _reserve_sha = await _redis.script_load(_RESERVE)
        _adjust_sha = await _redis.script_load(_ADJUST)
    return _redis


async def close() -> None:
    global _redis
    if _redis is not None:
        await _redis.aclose()
        _redis = None


def client() -> aioredis.Redis:
    if _redis is None:
        raise RuntimeError("limits.connect() has not run")
    return _redis


def _budget_key(tenant_id: int) -> str:
    return f"budget:{tenant_id}:{time.strftime('%Y-%m')}"


# ------------------------------------------------------------------- api ---
async def allow_request(tenant_id: int, rate: float, burst: int) -> tuple[bool, float]:
    allowed, remaining = await client().evalsha(
        _bucket_sha, 1, f"rl:{tenant_id}", rate, burst, time.time(), 1
    )
    return bool(allowed), float(remaining)


async def reserve_budget(tenant_id: int, budget: Decimal,
                         estimate: Decimal) -> tuple[bool, Decimal]:
    ok, spent = await client().evalsha(
        _reserve_sha, 1, _budget_key(tenant_id),
        str(budget), str(estimate), _MONTH_TTL,
    )
    return bool(ok), Decimal(spent)


async def settle_budget(tenant_id: int, estimate: Decimal, actual: Decimal) -> None:
    """Replace the estimate with the truth. Usually a refund, since max_tokens
    is an upper bound and completions are shorter than it."""
    delta = actual - estimate
    if delta != 0:
        await client().evalsha(_adjust_sha, 1, _budget_key(tenant_id),
                               str(delta), _MONTH_TTL)


async def refund_budget(tenant_id: int, estimate: Decimal) -> None:
    """A cache hit costs nothing, so the whole pre-authorisation goes back."""
    await client().evalsha(_adjust_sha, 1, _budget_key(tenant_id),
                           str(-estimate), _MONTH_TTL)


async def spend_so_far(tenant_id: int) -> Decimal:
    return Decimal(await client().get(_budget_key(tenant_id)) or "0")


# ------------------------------------------------------- single flight ----
# Cache stampede: 100 identical requests arriving on a cold entry all miss, and
# all 100 call the provider. Because the cache write is asynchronous (§25) the
# window is WIDER here than in a normal cache -- it stays open until Celery
# lands the row, not just until the first response returns.
#
# Fix: the first request to arrive takes a lock and calls the provider; the rest
# wait for it to publish the result. The holder publishes to Redis directly
# rather than waiting on the Celery write, so the two paths stay decoupled.

FILL_LOCK_TTL = 30          # seconds; must exceed provider timeout
FILL_RESULT_TTL = 60
FILL_POLL_INTERVAL = 0.05
FILL_WAIT_BUDGET = 2.0      # cap: never wait longer than a normal miss


async def acquire_fill_lock(key: str) -> bool:
    """SET NX is the whole lock. Returns True if WE are the filler."""
    return bool(await client().set(f"fill:{key}", "1", nx=True, ex=FILL_LOCK_TTL))


async def publish_fill_result(key: str, payload: str) -> None:
    r = client()
    await r.set(f"fillres:{key}", payload, ex=FILL_RESULT_TTL)
    await r.delete(f"fill:{key}")


async def release_fill_lock(key: str) -> None:
    """Called when the fill FAILED. Without this the lock lingers for its TTL
    and every waiter blocks on a request that is never coming."""
    await client().delete(f"fill:{key}")


async def await_fill_result(key: str) -> str | None:
    """Poll for the holder's result. Returns None on timeout, and the caller
    then proceeds to call the provider itself -- a duplicate call is far better
    than an unbounded wait on a holder that may have died."""
    import asyncio
    deadline = time.monotonic() + FILL_WAIT_BUDGET
    while time.monotonic() < deadline:
        payload = await client().get(f"fillres:{key}")
        if payload is not None:
            return payload
        await asyncio.sleep(FILL_POLL_INTERVAL)
    return None
```

> **Does** — rate limiting, budget reserve/settle, and the single-flight lock. All of it in Redis.
> **Watch out** — **everything here is a Lua script, and that is the whole point.** `GET`, compute in
> Python, `SET` back is a race: two concurrent requests read the same token count and both proceed, so
> a 10 rps limit passes 20. Redis executes Lua atomically against a single-threaded loop, so the check
> and the decrement cannot interleave. `tests/test_limits.py` fires 50 concurrent requests at a
> burst-10 bucket and asserts exactly 10 pass — a read-modify-write version passes the sequential test
> and fails that one.

---

## 21. `app/breaker.py`

```python
"""Circuit breaker with a provider failover chain.

State lives in REDIS, not in the process. With three API replicas an in-process
breaker means each one has to independently discover the provider is down --
three times the failed requests, and three different opinions about whether the
dependency is healthy. Shared state means the first replica to notice protects
the rest.

The state machine is deliberately simple:

    CLOSED     normal. Consecutive failures accumulate.
    OPEN       >= threshold failures. Every call short-circuits until cooldown.
    HALF-OPEN  cooldown elapsed. The next call is allowed through as a probe;
               success closes the circuit, failure re-opens it.

HALF-OPEN is implicit rather than a stored state: once `open_until` has passed,
`allow()` returns True again, so the next caller IS the probe. Under high
concurrency several probes may slip through together. That is a deliberate
trade -- a strictly single-probe implementation needs another lock, and letting
three requests test a recovering provider is cheap.
"""

from __future__ import annotations

import time

from . import limits
from .config import BREAKER_COOLDOWN_SECONDS, BREAKER_FAILURE_THRESHOLD
from .providers.base import Completion, Provider, ProviderError


def _fail_key(name: str) -> str:
    return f"breaker:{name}:failures"


def _open_key(name: str) -> str:
    return f"breaker:{name}:open_until"


async def is_open(name: str) -> bool:
    raw = await limits.client().get(_open_key(name))
    return raw is not None and time.time() < float(raw)


async def record_success(name: str) -> None:
    r = limits.client()
    await r.delete(_fail_key(name), _open_key(name))


async def record_failure(name: str) -> int:
    r = limits.client()
    failures = await r.incr(_fail_key(name))
    await r.expire(_fail_key(name), BREAKER_COOLDOWN_SECONDS * 4)
    if failures >= BREAKER_FAILURE_THRESHOLD:
        await r.set(_open_key(name), str(time.time() + BREAKER_COOLDOWN_SECONDS),
                    ex=BREAKER_COOLDOWN_SECONDS + 5)
    return failures


class AllProvidersDown(RuntimeError):
    pass


async def call_with_failover(chain: list[Provider], messages: list[dict],
                             model: str, temperature: float,
                             max_tokens: int) -> tuple[Completion, list[str]]:
    """Walk the chain until one succeeds. Returns (completion, providers_tried).

    A non-retryable ProviderError -- a 4xx, meaning WE sent something malformed --
    does not count against the breaker and is raised immediately. Opening the
    circuit on a healthy provider because of our own bad request would turn a
    400 into an outage.
    """
    tried: list[str] = []
    last: Exception | None = None

    for provider in chain:
        if await is_open(provider.name):
            tried.append(f"{provider.name}:open")
            continue
        tried.append(provider.name)
        try:
            completion = await provider.complete(messages, model, temperature, max_tokens)
        except ProviderError as e:
            if not e.retryable:
                raise
            await record_failure(provider.name)
            last = e
            continue
        except Exception as e:                      # timeout, transport, parse
            await record_failure(provider.name)
            last = e
            continue
        await record_success(provider.name)
        return completion, tried

    raise AllProvidersDown(f"chain exhausted {tried}: {last}")
```

> **Does** — the circuit breaker and `call_with_failover()`, which walks the provider chain.
> **Watch out** — state lives in **Redis, not the process**. With three API replicas an in-process
> breaker means each one independently rediscovers the outage: three times the failed requests and
> three different opinions about whether the dependency is healthy. HALF-OPEN is implicit — once
> `open_until` passes, the next caller *is* the probe — which lets a few probes slip through together
> under load. That is a deliberate trade against needing another lock.

---

## 22. `app/accounting.py`

```python
"""Cost estimation and the usage ledger."""

from __future__ import annotations

import uuid
from decimal import Decimal

from . import db
from .providers.base import Usage, cost_usd


def estimate(model: str, messages: list[dict], max_tokens: int) -> Decimal:
    """Upper bound, used to PRE-AUTHORISE against the budget.

    Deliberately pessimistic: assume the completion runs to max_tokens. A tenant
    at their cap should be stopped by the worst case, not sail past it on an
    optimistic guess and overshoot. settle_budget() refunds the difference
    moments later.

    ~4 characters per token is a rough English approximation -- good enough for a
    reservation, never used for the reported number.
    """
    chars = sum(len(m["content"]) for m in messages)
    return cost_usd(model, Usage(prompt_tokens=chars // 4, completion_tokens=max_tokens))


async def record(request_id: uuid.UUID, tenant_id: int, cache_hit: bool,
                 provider: str | None, model: str | None, usage: Usage,
                 cost: Decimal, latency_ms: int) -> None:
    """Append to the usage ledger.

    ON CONFLICT DO NOTHING against the UNIQUE request_id makes this idempotent:
    a Celery retry after a partial failure cannot double-count spend. The
    constraint enforces it, rather than us hoping the task runs exactly once.
    """
    await db.pool().execute(
        """
        INSERT INTO usage_events
            (request_id, tenant_id, cache_hit, provider, model,
             prompt_tokens, completion_tokens, cost_usd, latency_ms)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        ON CONFLICT (request_id) DO NOTHING
        """,
        request_id, tenant_id, cache_hit, provider, model,
        usage.prompt_tokens, usage.completion_tokens, cost, latency_ms,
    )
```

> **Does** — `estimate()` for the budget pre-authorisation, and `record()` for the usage ledger.
> **Watch out** — the estimate is **pessimistic on purpose**: it assumes the completion runs to
> `max_tokens`. A tenant at their cap should be stopped by the worst case, not sail past on an
> optimistic guess. `record()` is idempotent by *database constraint* — `ON CONFLICT DO NOTHING`
> against a UNIQUE `request_id` — rather than by hoping the Celery task runs exactly once.

---

## 23. `app/telemetry.py`

```python
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
```

> **Does** — defines every Prometheus metric. Counters for requests, cache outcomes, provider calls
> and cost; histograms for end-to-end and per-stage latency.
> **Watch out** — the histogram buckets are chosen **for this system**, not left at the library
> default. A cache hit lands near 16 ms and a provider call near 900 ms; the default buckets put both
> in one giant bin and the histogram tells you nothing. `bench/report.py` reads these.

---

## 24. `app/auth.py`

```python
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
```

> **Does** — resolves an `Authorization: Bearer` key to a `Tenant` row.
> **Watch out** — SHA-256, **not bcrypt**, and that is deliberate. bcrypt is intentionally slow
> (~100 ms), which is correct for human-chosen passwords and wrong for a header checked on every
> request — it would dominate the entire pre-provider budget on its own. API keys are high-entropy
> random strings, so they are not brute-forceable the way a password is.

---

## 25. `app/tasks.py`

```python
"""Celery: everything that must NOT be on the request path.

Writing the cache entry is the single most tempting thing to do synchronously
and the single worst. It adds a Postgres insert plus an embedding write to every
miss, on the path the caller is waiting on, purely to benefit some FUTURE
request. Moving it here costs one thing: two identical requests arriving within
a few hundred milliseconds of each other both miss. That is a good trade, and
being able to say so out loud is the point.
"""

from __future__ import annotations

import asyncio
import uuid
from decimal import Decimal

import numpy as np
from celery import Celery

from . import accounting, db
from .cache import semantic
from .config import REDIS_URL
from .providers.base import Usage

celery_app = Celery("gateway", broker=REDIS_URL, backend=None)
celery_app.conf.update(
    task_acks_late=True,              # redeliver if a worker dies mid-task
    worker_prefetch_multiplier=1,     # do not hoard tasks into one worker
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    beat_schedule={
        # TTL is enforced at READ time, so expired rows are filtered out but
        # never removed. Without this the table and its index grow forever and
        # the exact-KNN scan degrades -- slowly, invisibly, in production.
        "prune-expired-cache": {"task": "prune_expired_cache", "schedule": 3600.0},
    },
)


def _run(coro):
    """Celery workers are synchronous; our data layer is async. One event loop
    per task keeps them from sharing a pool across threads."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(db.connect())
        return loop.run_until_complete(coro)
    finally:
        loop.run_until_complete(db.close())
        loop.close()


@celery_app.task(name="persist_cache_entry", max_retries=3,
                 autoretry_for=(Exception,), retry_backoff=True)
def persist_cache_entry(tenant_id: int, ns: str, prompt: str, vector: list,
                        response: str, prompt_tokens: int,
                        completion_tokens: int, provider: str, model: str):
    _run(semantic.store(tenant_id, ns, prompt,
                        np.asarray(vector, dtype=np.float32), response,
                        prompt_tokens, completion_tokens, provider, model))


@celery_app.task(name="prune_expired_cache")
def prune_expired_cache():
    """Delete cache entries past TTL. Runs hourly via Celery beat.

    Deletes in BATCHES rather than one statement. A single unbounded DELETE takes
    a long transaction and holds locks that block inserts on the hot path -- the
    cleanup job would cause the outage it exists to prevent.
    """
    return _run(_prune())


async def _prune(batch: int = 5_000) -> int:
    from .config import CACHE_TTL_HOURS
    total = 0
    while True:
        # execute() returns a status string -- "DELETE 4231" -- so we learn how
        # many rows actually went. fetchval() would only tell us whether ANY did,
        # and the count would be a guess.
        status = await db.pool().execute(
            """
            DELETE FROM cache_entries
            WHERE id IN (
                SELECT id FROM cache_entries
                WHERE created_at < now() - ($1 || ' hours')::interval
                LIMIT $2
            )
            """,
            str(CACHE_TTL_HOURS), batch,
        )
        deleted = int(status.rsplit(" ", 1)[-1])
        total += deleted
        if deleted < batch:              # partial batch means we caught up
            break
        if total >= 1_000_000:           # safety valve, resume next hour
            break
    return total


@celery_app.task(name="record_usage", max_retries=5,
                 autoretry_for=(Exception,), retry_backoff=True)
def record_usage(request_id: str, tenant_id: int, cache_hit: bool,
                 provider: str | None, model: str | None, prompt_tokens: int,
                 completion_tokens: int, cost_usd: str, latency_ms: int):
    # Retries are safe: usage_events.request_id is UNIQUE and the insert is
    # ON CONFLICT DO NOTHING (§22).
    _run(accounting.record(
        uuid.UUID(request_id), tenant_id, cache_hit, provider, model,
        Usage(prompt_tokens, completion_tokens), Decimal(cost_usd), latency_ms,
    ))
```

> **Does** — everything that must not block the caller: persisting cache entries, writing the usage
> ledger, and the hourly TTL reaper.
> **Watch out** — the reaper deletes in **batches**. A single unbounded `DELETE` holds locks that
> block inserts on the hot path, so the cleanup job would cause the outage it exists to prevent. Also:
> `beat` is a separate process from `worker`, and **exactly one of it may ever run** — two beat
> processes double-schedule every periodic task.

---

## 26. `app/main.py`

```python
"""The request path.

Read this top to bottom and the whole design is visible: every step before the
provider call costs about 16 ms in total, and everything that does not have to
block the caller happens after the response is already on its way back.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.concurrency import run_in_threadpool

from . import accounting, breaker, db, limits, tasks, telemetry
from .auth import Tenant, resolve
from .cache import embed, semantic
from .config import DEFAULT_MODEL, PROVIDER_CHAIN
from .models import ChatRequest, ChatResponse
from .providers.base import Usage, cost_usd
from .providers.gemini import GeminiProvider
from .providers.groq import GroqProvider
from .providers.mock import MockProvider

_REGISTRY = {"mock": MockProvider, "groq": GroqProvider, "gemini": GeminiProvider}
CHAIN = [_REGISTRY[name]() for name in PROVIDER_CHAIN if name in _REGISTRY]


@asynccontextmanager
async def lifespan(_: FastAPI):
    await db.connect()
    await limits.connect()
    # Eager load: a lazy first encode costs 2-3 s. Paying it here means the
    # container is not healthy until it can actually serve, instead of
    # accepting traffic and putting 3 s into someone's p99.
    await run_in_threadpool(embed.load)
    yield
    await db.close()
    await limits.close()


app = FastAPI(title="LLM Gateway", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    await db.pool().fetchval("SELECT 1")
    await limits.client().ping()
    return {"ok": True, "providers": [p.name for p in CHAIN]}


@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, tenant: Tenant = Depends(resolve)) -> ChatResponse:
    t0 = time.perf_counter()
    request_id = uuid.uuid4()
    model = req.model or DEFAULT_MODEL
    messages = [m.model_dump() for m in req.messages]

    # -- 1. rate limit ------------------------------------------------ ~1 ms
    allowed, remaining = await limits.allow_request(
        tenant.id, tenant.rate_per_sec, tenant.burst)
    if not allowed:
        telemetry.REQUESTS.labels(tenant.name, "rate_limited").inc()
        raise HTTPException(429, "rate limit exceeded",
                            headers={"Retry-After": "1"})

    # -- 2. budget pre-authorisation ---------------------------------- ~1 ms
    estimate = accounting.estimate(model, messages, req.max_tokens)
    within, spent = await limits.reserve_budget(
        tenant.id, tenant.monthly_budget_usd, estimate)
    if not within:
        telemetry.REQUESTS.labels(tenant.name, "over_budget").inc()
        # 402 rather than 429: retrying will not help, the money is gone.
        raise HTTPException(402, f"monthly budget exhausted (spent ${spent})")

    # -- 3. semantic cache -------------------------------------------- ~13 ms
    prompt = semantic.query_text(messages)
    vector = None
    hit = None
    eligible = not req.no_cache and semantic.is_eligible(req.temperature, prompt)

    if eligible:
        with telemetry.STAGE.labels("embed").time():
            # encode() is CPU-bound and BLOCKING. Calling it directly on the
            # event loop stalls every other in-flight request for its duration --
            # 5 ms x 200 concurrent requests serialises into a 1 s queue. The
            # threadpool keeps the loop free.
            vector = await run_in_threadpool(embed.encode, prompt)
        ns = semantic.namespace(model, messages, req.temperature)
        with telemetry.STAGE.labels("cache_lookup").time():
            hit = await semantic.lookup(tenant.id, ns, prompt, vector)
    else:
        telemetry.CACHE.labels("skip").inc()

    if hit is not None:
        await limits.refund_budget(tenant.id, estimate)   # a hit costs nothing
        latency_ms = int((time.perf_counter() - t0) * 1000)
        telemetry.CACHE.labels("hit").inc()
        telemetry.REQUESTS.labels(tenant.name, "cached").inc()
        telemetry.LATENCY.labels("true").observe(latency_ms / 1000)
        tasks.record_usage.delay(
            str(request_id), tenant.id, True, hit.provider, hit.model,
            hit.prompt_tokens, hit.completion_tokens, "0", latency_ms)
        return ChatResponse(
            id=str(request_id), content=hit.response, model=hit.model,
            provider=hit.provider, cached=True, similarity=hit.similarity,
            prompt_tokens=hit.prompt_tokens,
            completion_tokens=hit.completion_tokens,
            cost_usd=0.0, latency_ms=latency_ms)

    if eligible:
        telemetry.CACHE.labels("miss").inc()

    # -- 4. single flight ---------------------------------------------------
    # Without this, N identical concurrent requests all miss and all pay. The
    # first takes the lock and calls the provider; the rest wait for its result
    # for at most FILL_WAIT_BUDGET, then give up and call anyway.
    fill_key = None
    if eligible:
        fill_key = hashlib.sha256(
            f"{semantic.namespace(model, messages, req.temperature)}\x00{prompt}".encode()
        ).hexdigest()[:32]
        if not await limits.acquire_fill_lock(fill_key):
            payload = await limits.await_fill_result(fill_key)
            if payload is not None:
                shared = json.loads(payload)
                await limits.refund_budget(tenant.id, estimate)
                latency_ms = int((time.perf_counter() - t0) * 1000)
                telemetry.CACHE.labels("coalesced").inc()
                telemetry.LATENCY.labels("true").observe(latency_ms / 1000)
                return ChatResponse(
                    id=str(request_id), content=shared["text"],
                    model=shared["model"], provider=shared["provider"],
                    cached=True, similarity=1.0,
                    prompt_tokens=shared["prompt_tokens"],
                    completion_tokens=shared["completion_tokens"],
                    cost_usd=0.0, latency_ms=latency_ms)
            fill_key = None          # holder died or was too slow; fill it ourselves

    # -- 5. the provider ---------------------------------------------- ~900 ms
    try:
        with telemetry.STAGE.labels("provider").time():
            completion, tried = await breaker.call_with_failover(
                CHAIN, messages, model, req.temperature, req.max_tokens)
    except breaker.AllProvidersDown as e:
        if fill_key:
            # Release, or every waiter blocks for the full lock TTL on a
            # request that already failed.
            await limits.release_fill_lock(fill_key)
        await limits.refund_budget(tenant.id, estimate)
        telemetry.REQUESTS.labels(tenant.name, "upstream_down").inc()
        raise HTTPException(503, f"all providers unavailable: {e}") from e

    if fill_key:
        await limits.publish_fill_result(fill_key, json.dumps({
            "text": completion.text, "model": completion.model,
            "provider": completion.provider,
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
        }))

    for name in tried:
        telemetry.PROVIDER_CALLS.labels(name, "attempt").inc()

    # -- 6. settle and reply ------------------------------------------------
    actual = cost_usd(completion.model, completion.usage)
    await limits.settle_budget(tenant.id, estimate, actual)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    telemetry.REQUESTS.labels(tenant.name, "ok").inc()
    telemetry.LATENCY.labels("false").observe(latency_ms / 1000)
    telemetry.COST.labels(tenant.name, completion.model).inc(float(actual))

    # -- 7. off the hot path ------------------------------------------------
    if eligible and vector is not None:
        tasks.persist_cache_entry.delay(
            tenant.id, semantic.namespace(model, messages, req.temperature),
            prompt, vector.tolist(), completion.text,
            completion.usage.prompt_tokens, completion.usage.completion_tokens,
            completion.provider, completion.model)

    tasks.record_usage.delay(
        str(request_id), tenant.id, False, completion.provider, completion.model,
        completion.usage.prompt_tokens, completion.usage.completion_tokens,
        str(actual), latency_ms)

    return ChatResponse(
        id=str(request_id), content=completion.text, model=completion.model,
        provider=completion.provider, cached=False, similarity=None,
        prompt_tokens=completion.usage.prompt_tokens,
        completion_tokens=completion.usage.completion_tokens,
        cost_usd=float(actual), latency_ms=latency_ms)


@app.get("/v1/usage")
async def usage(tenant: Tenant = Depends(resolve)):
    """What every finance team actually asks for: who spent what."""
    spent = await limits.spend_so_far(tenant.id)
    rows = await db.pool().fetch(
        """
        SELECT date_trunc('hour', created_at) AS hour,
               count(*) AS requests,
               count(*) FILTER (WHERE cache_hit) AS cache_hits,
               sum(cost_usd) AS cost_usd
        FROM usage_events
        WHERE tenant_id = $1 AND created_at > now() - interval '24 hours'
        GROUP BY 1 ORDER BY 1 DESC
        """, tenant.id)
    return {
        "tenant": tenant.name,
        "month_to_date_usd": float(spent),
        "budget_usd": float(tenant.monthly_budget_usd),
        "hourly": [dict(r) for r in rows],
    }
```

> **Does** — the request path, and therefore the design. Read it top to bottom and the whole system
> is visible.
> **Watch out** — **the order of the seven steps is the architecture.** Auth, rate limit and budget
> come first because they are cheap and can reject early. The embedding and cache lookup come before
> the provider because that is the entire point. Single-flight sits between the miss and the provider
> so N identical cold requests become one call. Everything after the response — cache write, cost
> accounting — is enqueued, never awaited. Every exit path either settles or refunds the budget
> reservation; miss one and the counter drifts from the ledger.

---

## 27. `tests/`

Five files. One of them is the reason this project can be trusted; the rest are ordinary.

### `tests/conftest.py`

```python
import os

import pytest
import pytest_asyncio

os.environ.setdefault("PROVIDER_CHAIN", "mock")


@pytest_asyncio.fixture
async def redis_client():
    from app import limits
    try:
        await limits.connect()
        await limits.client().ping()
    except Exception:                                   # noqa: BLE001
        pytest.skip("redis unavailable -- run `docker compose up -d redis`")
    await limits.client().flushdb()
    yield limits.client()
    await limits.close()
```

> **Does** — one fixture: a Redis client that flushes between tests.
> **Watch out** — it **skips** rather than fails when Redis is unavailable, so
> `tests/test_cache_correctness.py` still runs on a bare checkout with no infrastructure. That is
> deliberate: the tests that matter most should need the least to run.

### `tests/test_cache_correctness.py` — the important one

```python
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
```

> **Does** — the test the whole project rests on. Ten adversarial pairs that must never hit, six
> paraphrases that must, the guard's individual rules, and the namespace/eligibility guards.
> **Watch out** — every similarity in the comments is **measured**, not assumed. Two tests are
> unusual and intentional: `test_threshold_alone_is_insufficient` pins the *finding* — it fails if a
> future embedding model separates the classes, which would mean the lexical guard has become dead
> code worth removing. And `test_guard_is_what_saves_the_high_similarity_pairs` asserts that the
> three pairs above threshold are caught by the guard rather than by the number, so nobody can
> "fix" a failure by nudging the threshold.

### `tests/test_limits.py`

```python
"""Proves the Lua script is actually atomic. This is the test that would fail
against a read-modify-write implementation."""

import asyncio
from decimal import Decimal

import pytest

from app import limits


async def test_bucket_allows_exactly_burst_then_refuses(redis_client):
    allowed = 0
    for _ in range(30):
        ok, _ = await limits.allow_request(tenant_id=1, rate=0.001, burst=10)
        allowed += ok
    assert allowed == 10


async def test_concurrent_requests_cannot_exceed_burst(redis_client):
    """THE concurrency test. Read-then-write passes the sequential test above and
    fails this one -- 50 coroutines all read the same token count and all decide
    they have room."""
    results = await asyncio.gather(*[
        limits.allow_request(tenant_id=2, rate=0.001, burst=10) for _ in range(50)
    ])
    assert sum(ok for ok, _ in results) == 10


async def test_bucket_refills_over_time(redis_client):
    for _ in range(5):
        await limits.allow_request(3, rate=100.0, burst=5)
    ok, _ = await limits.allow_request(3, rate=100.0, burst=5)
    assert not ok
    await asyncio.sleep(0.1)                      # 100/s * 0.1 s = 10 tokens
    ok, _ = await limits.allow_request(3, rate=100.0, burst=5)
    assert ok


async def test_budget_reserve_settle_and_refund(redis_client):
    budget = Decimal("1.00")
    ok, spent = await limits.reserve_budget(9, budget, Decimal("0.40"))
    assert ok and spent == Decimal("0.40")

    await limits.settle_budget(9, Decimal("0.40"), Decimal("0.10"))   # refund 0.30
    assert await limits.spend_so_far(9) == Decimal("0.10")

    ok, _ = await limits.reserve_budget(9, budget, Decimal("0.95"))
    assert ok                                     # 0.10 < 1.00, reservation allowed
    ok, _ = await limits.reserve_budget(9, budget, Decimal("0.10"))
    assert not ok                                 # now 1.05 >= 1.00, refused


async def test_cache_hit_refunds_the_whole_reservation(redis_client):
    await limits.reserve_budget(11, Decimal("5"), Decimal("0.25"))
    await limits.refund_budget(11, Decimal("0.25"))
    assert await limits.spend_so_far(11) == Decimal("0")


# ------------------------------------------------------- single flight ----
async def test_only_one_caller_wins_the_fill_lock(redis_client):
    """The stampede guard. 20 coroutines race for the same cold key; exactly one
    may call the provider."""
    got = await asyncio.gather(*[
        limits.acquire_fill_lock("same-key") for _ in range(20)
    ])
    assert sum(got) == 1


async def test_waiter_receives_the_holders_result(redis_client):
    assert await limits.acquire_fill_lock("k1")

    async def holder():
        await asyncio.sleep(0.1)
        await limits.publish_fill_result("k1", '{"text": "shared"}')

    async def waiter():
        return await limits.await_fill_result("k1")

    _, payload = await asyncio.gather(holder(), waiter())
    assert payload is not None and "shared" in payload


async def test_waiter_gives_up_rather_than_hanging(redis_client):
    """If the holder dies, waiters must fall through to calling the provider
    themselves. A duplicate call beats an unbounded wait."""
    assert await limits.acquire_fill_lock("k2")
    limits.FILL_WAIT_BUDGET = 0.2                     # keep the test fast
    assert await limits.await_fill_result("k2") is None


async def test_releasing_the_lock_lets_the_next_caller_fill(redis_client):
    """A failed fill must release, or every waiter blocks for the full lock TTL
    on a request that already failed."""
    assert await limits.acquire_fill_lock("k3")
    assert not await limits.acquire_fill_lock("k3")
    await limits.release_fill_lock("k3")
    assert await limits.acquire_fill_lock("k3")
```

> **Does** — proves the Lua scripts are atomic, the bucket refills, the budget reserve/settle cycle
> balances, and the single-flight lock elects exactly one filler.
> **Watch out** — `test_concurrent_requests_cannot_exceed_burst` is the one that matters. A
> read-modify-write implementation **passes the sequential test above it and fails that one**. If you
> ever rewrite `limits.py` in Python for readability, this is the test that stops you.

### `tests/test_breaker.py`

```python
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
```

> **Does** — the circuit opens at threshold, an open circuit short-circuits without calling the
> provider, failover reaches the second one, and success clears prior failures.
> **Watch out** — `test_our_own_bad_request_does_not_open_the_circuit` encodes the rule that a 4xx
> must not count as a provider failure. Without it, one malformed request from your own code takes a
> healthy provider offline for every tenant.

### `tests/test_accounting.py`

```python
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
```

> **Does** — cost arithmetic is exact `Decimal`, an unknown model costs zero rather than crashing,
> and the budget estimate is pessimistic.
> **Watch out** — `test_cost_is_exact_decimal_not_float` looks pedantic and is not. This number is
> summed across millions of rows and compared against a spend cap; float drift there is a bug you
> find in a billing dispute.

---

## 28. `bench/`

### `bench/threshold_sweep.py` — the headline experiment

```python
"""Sweep the similarity threshold and show why a threshold alone is not enough.

Produces bench/results/threshold_curve.png and threshold.json, and prints the
project's central finding: the adversarial and paraphrase similarity
distributions OVERLAP, so no cut point separates them (DESIGN.md §3).

    python bench/threshold_sweep.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Runnable as `python bench/threshold_sweep.py` from a checkout, not only after
# `pip install -e .`. Two lines that remove a papercut for anyone cloning this.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

from app.cache.embed import encode           # noqa: E402
from app.cache.semantic import lexical_conflict  # noqa: E402
from tests.test_cache_correctness import ADVERSARIAL, PARAPHRASE  # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)


def sims(pairs):
    return np.array([float(np.dot(encode(a), encode(b))) for a, b in pairs])


def main():
    ok = sims(PARAPHRASE)          # should hit
    bad = sims(ADVERSARIAL)        # must not hit
    guarded = np.array([lexical_conflict(a, b) is not None for a, b in ADVERSARIAL])

    grid = np.linspace(0.60, 0.999, 120)
    useful = [(ok >= t).mean() for t in grid]
    wrong_cosine = [(bad >= t).mean() for t in grid]
    # With the guard, an adversarial pair only counts as a wrong hit if it is
    # above threshold AND the lexical check failed to catch it.
    wrong_guarded = [((bad >= t) & ~guarded).mean() for t in grid]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(grid, useful, label="useful hit rate (paraphrases)", color="steelblue")
    ax.plot(grid, wrong_cosine, label="WRONG hits — cosine only", color="crimson")
    ax.plot(grid, wrong_guarded, label="WRONG hits — cosine + lexical guard",
            color="seagreen", ls="--", lw=2)
    ax.axvspan(bad.max(), ok.max(), alpha=0.12, color="crimson")
    ax.annotate("overlap: no threshold\nseparates the classes",
                xy=((bad.max() + ok.max()) / 2, 0.55), ha="center", fontsize=9)
    ax.set_xlabel("cosine similarity threshold")
    ax.set_ylabel("rate")
    ax.set_title("A threshold alone cannot make a semantic cache safe")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "threshold_curve.png", dpi=150)

    overlaps = bool(bad.max() > ok.min())
    result = {
        "paraphrase": {"min": float(ok.min()), "max": float(ok.max())},
        "adversarial": {"min": float(bad.min()), "max": float(bad.max())},
        "distributions_overlap": overlaps,
        "margin": float(ok.min() - bad.max()),
        "adversarial_caught_by_guard": int(guarded.sum()),
        "adversarial_total": len(ADVERSARIAL),
    }
    (OUT / "threshold.json").write_text(json.dumps(result, indent=2))

    print(f"paraphrase   min={ok.min():.4f}  max={ok.max():.4f}")
    print(f"adversarial  min={bad.min():.4f}  max={bad.max():.4f}")
    print(f"margin (paraphrase floor - adversarial ceiling) = {result['margin']:+.4f}")

    if overlaps:
        print("\nFINDING: the distributions OVERLAP.")
        print(f"  a WRONG pair scores {bad.max():.4f}")
        print(f"  a RIGHT pair scores {ok.min():.4f}")
        print("  -> no threshold admits real paraphrases and rejects the wrong pairs.")
        print(f"  -> the lexical guard catches {guarded.sum()}/{len(ADVERSARIAL)} "
              f"adversarial pairs, including every one above threshold.")
    else:
        print("\nThe distributions separate cleanly; a threshold alone would suffice. "
              "Re-check whether the guard is still earning its place.")


if __name__ == "__main__":
    main()
```

> **Does** — the headline experiment. Sweeps the similarity threshold over the labelled pairs and
> shows that **no threshold separates the classes**, then shows the lexical guard closing the gap.
> **Watch out** — the output to quote is the **margin**: the gap between the worst paraphrase and the
> best adversarial pair. It is negative, which is the finding. If a future embedding model makes it
> positive, the guard has become unnecessary and that is worth knowing rather than silently carrying.

### `bench/cost_model.py`

```python
"""Cost per 1,000 requests as a function of cache hit rate, with the break-even
point marked. This is the plot that answers 'why does this service exist'."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

from app.config import PRICES     # noqa: E402

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

MODEL = "llama-3.3-70b-versatile"
PROMPT_TOKENS, COMPLETION_TOKENS = 800, 250
MONTHLY_REQUESTS = 2_000_000

# Gateway infrastructure: 1 small VM + managed Postgres + Redis, roughly.
GATEWAY_FIXED_USD_PER_MONTH = 25.0


def main():
    p_price, c_price = PRICES[MODEL]
    per_call = float(PROMPT_TOKENS * p_price + COMPLETION_TOKENS * c_price) / 1e6

    hit = np.linspace(0, 0.8, 81)
    per_1k_direct = np.full_like(hit, per_call * 1000)
    per_1k_gateway = per_call * 1000 * (1 - hit) + \
        GATEWAY_FIXED_USD_PER_MONTH / (MONTHLY_REQUESTS / 1000)

    break_even = hit[np.argmax(per_1k_gateway <= per_1k_direct)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hit * 100, per_1k_direct, ls=":", color="gray", label="direct to provider")
    ax.plot(hit * 100, per_1k_gateway, label="through the gateway")
    ax.axvline(break_even * 100, ls="--", color="k",
               label=f"break-even = {break_even*100:.1f}% hit rate")
    ax.set_xlabel("cache hit rate (%)")
    ax.set_ylabel("USD per 1,000 requests")
    ax.set_title(f"Cost vs cache hit rate — {MODEL}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cost_curve.png", dpi=150)

    for h in (0.0, 0.2, 0.4, 0.6):
        monthly = per_call * MONTHLY_REQUESTS * (1 - h) + GATEWAY_FIXED_USD_PER_MONTH
        print(f"hit rate {h:>4.0%}  ->  ${monthly:>9,.2f}/month at "
              f"{MONTHLY_REQUESTS:,} requests")


if __name__ == "__main__":
    main()
```

> **Does** — cost per 1,000 requests as a function of cache hit rate, with the break-even against
> calling the provider directly.
> **Watch out** — this is **arithmetic on published prices**, not a bill. Say so in the README. The
> real per-request cost lives in `usage_events` and is only meaningful if you ran against a real
> provider.

### `bench/load.js`

```javascript
// k6 load test. ALWAYS runs against PROVIDER_CHAIN=mock (DESIGN.md §5c):
// benchmarking a real provider measures THEIR latency variance, not ours.
//
//   k6 run -e API_KEY=demo-key-please-change bench/load.js
import http from 'k6/http';
import { check } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const cacheHits = new Rate('cache_hit_rate');
const hitLatency = new Trend('latency_cache_hit_ms');
const missLatency = new Trend('latency_cache_miss_ms');

const BASE = __ENV.BASE_URL || 'http://localhost:8000';
const KEY = __ENV.API_KEY;

// A small pool of questions produces repeats, which is what a real workload
// looks like. Unique-per-request traffic would measure the miss path only.
const POOL = [
  'What is the capital of France?', "What's the capital city of France?",
  'How do I reset my password?', 'How can I reset my password?',
  'Summarise this document', 'Give me a summary of this document',
  'When are you open?', 'What are your business hours?',
];

export const options = {
  scenarios: {
    warm: {                                   // realistic: repeats -> cache hits
      executor: 'ramping-arrival-rate',
      startRate: 10, timeUnit: '1s',
      preAllocatedVUs: 100, maxVUs: 400,
      stages: [
        { target: 50, duration: '30s' },
        { target: 200, duration: '1m' },
        { target: 200, duration: '2m' },
        { target: 0, duration: '30s' },
      ],
    },
  },
  thresholds: {
    'http_req_failed': ['rate<0.01'],
    'latency_cache_hit_ms': ['p(95)<50'],     // the cache path must stay fast
    'http_req_duration': ['p(95)<2500'],
  },
};

export default function () {
  const body = JSON.stringify({
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: POOL[Math.floor(Math.random() * POOL.length)] },
    ],
    temperature: 0.0,
    max_tokens: 128,
  });

  const res = http.post(`${BASE}/v1/chat`, body, {
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${KEY}` },
  });

  check(res, { 'status 200': (r) => r.status === 200 });
  if (res.status === 200) {
    const j = res.json();
    cacheHits.add(j.cached);
    (j.cached ? hitLatency : missLatency).add(j.latency_ms);
  }
}
```

> **Does** — generates load and records the numbers the README quotes.
> **Watch out** — two things earn their place. It records hit and miss latency into **separate
> Trends**, because the distribution is bimodal (16 ms and 900 ms) and a single average reports a
> latency no request ever experienced. And it uses `ramping-arrival-rate`, which holds a target
> req/s **regardless of how slow the server gets** — with fixed VUs, a slowing server receives less
> load and you never find the breaking point.

### `bench/report.py` — turn `/metrics` into the README tables

Every other number is handed to you by k6 or a bench script. The **stage breakdown** is not:
Prometheus exposes histograms as cumulative bucket counters, and getting a p95 out of them by hand
means interpolating between buckets. This does that and prints markdown ready to paste.

```python
"""Scrape /metrics and print the README tables.

    python bench/report.py                      # after a k6 run
    python bench/report.py --url http://api:8000

Prometheus histograms are CUMULATIVE bucket counters, not percentiles:

    gw_stage_latency_seconds_bucket{stage="embed",le="0.01"} 4823
    gw_stage_latency_seconds_bucket{stage="embed",le="0.025"} 5102

To get p95 you find the bucket containing the 95th-percentile rank and
interpolate linearly inside it -- the same thing PromQL's histogram_quantile
does, reimplemented here so the project needs no Prometheus server to report.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from collections import defaultdict

SAMPLE = re.compile(r"^(?P<name>[a-zA-Z_:][\w:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[^\s]+)$")
LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    out: dict[str, list[tuple[dict[str, str], float]]] = defaultdict(list)
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = SAMPLE.match(line)
        if not m:
            continue
        labels = dict(LABEL.findall(m.group("labels") or ""))
        try:
            value = float(m.group("value"))
        except ValueError:
            continue                      # NaN on an untouched counter
        out[m.group("name")].append((labels, value))
    return out


def quantile(buckets: list[tuple[float, float]], q: float) -> float | None:
    """`buckets` is [(le, cumulative_count)] sorted ascending, +Inf last.

    Returns None when the histogram is empty -- an honest "no data" beats a
    confident 0.0 that looks like a measurement.
    """
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None

    rank = q * total
    prev_le, prev_count = 0.0, 0.0
    for le, count in buckets:
        if count >= rank:
            if le == float("inf"):
                return prev_le            # tail beyond the last finite bucket
            if count == prev_count:
                return le
            frac = (rank - prev_count) / (count - prev_count)
            return prev_le + (le - prev_le) * frac
        prev_le, prev_count = le, count
    return buckets[-1][0]


def histogram(samples, key: str, value: str) -> list[tuple[float, float]]:
    """Collect one label-slice of a histogram into sorted (le, count) pairs."""
    pairs = []
    for labels, v in samples:
        if labels.get(key) != value:
            continue
        le = labels.get("le")
        pairs.append((float("inf") if le == "+Inf" else float(le), v))
    return sorted(pairs)


def ms(x: float | None) -> str:
    return "—" if x is None else f"{x * 1000:.1f} ms"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/metrics")
    a = ap.parse_args()

    try:
        with urllib.request.urlopen(a.url, timeout=5) as r:
            text = r.read().decode()
    except Exception as e:                                    # noqa: BLE001
        print(f"could not read {a.url}: {e}", file=sys.stderr)
        print("is the stack up?  docker compose up -d", file=sys.stderr)
        return 1

    m = parse(text)

    # ---- stage breakdown --------------------------------------------------
    stage_samples = m.get("gw_stage_latency_seconds_bucket", [])
    stages = sorted({l.get("stage") for l, _ in stage_samples if l.get("stage")})
    print("### Where the latency goes\n")
    print("| Stage | p50 | p95 | p99 | count |")
    print("|---|---|---|---|---|")
    before_provider = 0.0
    for stage in stages:
        b = histogram(stage_samples, "stage", stage)
        n = b[-1][1] if b else 0
        p50, p95, p99 = (quantile(b, q) for q in (0.50, 0.95, 0.99))
        if stage != "provider" and p50:
            before_provider += p50
        print(f"| {stage} | {ms(p50)} | {ms(p95)} | {ms(p99)} | {n:,.0f} |")
    print(f"| **total before provider (p50)** | **{before_provider * 1000:.1f} ms** | | | |")

    # ---- end-to-end, split by cache outcome -------------------------------
    req = m.get("gw_request_latency_seconds_bucket", [])
    print("\n### End-to-end\n")
    print("| Path | p50 | p95 | p99 | count |")
    print("|---|---|---|---|---|")
    for cached, label in (("true", "cache hit"), ("false", "cache miss")):
        b = histogram(req, "cached", cached)
        n = b[-1][1] if b else 0
        print(f"| {label} | {ms(quantile(b, .50))} | {ms(quantile(b, .95))} "
              f"| {ms(quantile(b, .99))} | {n:,.0f} |")

    # ---- cache outcomes ---------------------------------------------------
    counts = {l.get("result"): v for l, v in m.get("gw_cache_total", [])}
    hit, miss = counts.get("hit", 0), counts.get("miss", 0)
    looked = hit + miss
    print("\n### Cache\n")
    print(f"- lookups: **{looked:,.0f}**  (hit {hit:,.0f} / miss {miss:,.0f})")
    print(f"- hit rate: **{(hit / looked * 100) if looked else 0:.1f}%**")
    for extra in ("skip", "coalesced"):
        if counts.get(extra):
            print(f"- {extra}: {counts[extra]:,.0f}")

    # ---- spend ------------------------------------------------------------
    cost = sum(v for _, v in m.get("gw_cost_usd_total", []))
    reqs = sum(v for _, v in m.get("gw_requests_total", []))
    print("\n### Cost\n")
    print(f"- total: **${cost:.4f}** over {reqs:,.0f} requests")
    if reqs:
        print(f"- per 1,000 requests: **${cost / reqs * 1000:.4f}**")
    print("\n> Derived from published provider pricing and measured token counts, "
          "not from a bill.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

> **Does** — scrapes `/metrics` and prints the README tables: stage breakdown, end-to-end latency
> split by cache outcome, hit rate, and spend.
> **Watch out** — Prometheus histograms are **cumulative bucket counters**, not percentiles, so this
> reimplements `histogram_quantile` — find the bucket containing the rank, interpolate linearly
> inside it. Run it **after** k6, not during: the counters are cumulative since process start, so a
> mid-run snapshot mixes warm-up with steady state.

Run it **after** k6, not during — the counters are cumulative since process start, so a mid-run
snapshot mixes warm-up with steady state. If you want a clean slate, `docker compose restart api`
first.

### `bench/chaos.sh`

```bash
#!/usr/bin/env bash
# Kill the primary provider mid-load and prove nothing 500s.
#
# Requires PROVIDER_CHAIN=mock,mock_backup with the primary configured to fail.
# We simulate the outage by flipping MOCK_FAILURE_RATE to 1.0 on the api
# container, which is the same thing from the gateway's point of view: every
# call to the primary raises.
set -euo pipefail

API=${BASE_URL:-http://localhost:8000}
KEY=${API_KEY:-demo-key-please-change}

echo "== baseline =="
curl -sf -XPOST "$API/v1/chat" -H "Authorization: Bearer $KEY" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"chaos probe"}],"no_cache":true}' \
  | python -c 'import sys,json; d=json.load(sys.stdin); print("provider:", d["provider"])'

echo "== starting load in background =="
k6 run -e API_KEY="$KEY" -e BASE_URL="$API" bench/load.js &
K6_PID=$!
sleep 20

echo "== killing the primary provider =="
docker compose exec -T api sh -c 'kill -TERM 1' || true    # or: docker compose stop provider-primary
sleep 5

echo "== during outage: requests must still succeed via failover =="
for i in $(seq 1 10); do
  code=$(curl -s -o /dev/null -w '%{http_code}' -XPOST "$API/v1/chat" \
    -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"probe '"$i"'"}],"no_cache":true}')
  echo "  probe $i -> HTTP $code"
done

wait $K6_PID
echo "== done: check http_req_failed in the k6 summary =="
```

> **Does** — kills the primary provider mid-load and checks that requests keep succeeding via
> failover.
> **Watch out** — success is not "it didn't crash". It is `http_req_failed` staying under 1% in the
> k6 summary **while** `gw_provider_calls_total` shows traffic shifting to the fallback. Check both,
> or you are only proving the breaker opened, not that anyone was served.

---

## 29. `Dockerfile`

```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf

WORKDIR /srv

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Copy the entire project FIRST
COPY . .

# Now app/ exists
RUN pip install --no-cache-dir -e ".[dev]"

# Download embedding model during build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --retries=3 \
CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

> **Does** — builds the image used by `api`, `worker` and `beat`.
> **Watch out** — the embedding model is **baked into the image**, not downloaded at runtime.
> Downloading 80 MB on first request makes a cold start look like a 3 s outage. The `HEALTHCHECK`
> matters too: compose uses it to order startup, and the app is not healthy until it can actually
> serve.

---

## 30. `docker-compose.yml`

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: gw
      POSTGRES_PASSWORD: gw
      POSTGRES_DB: gateway
    ports: ["5432:5432"]
    volumes:
      - pgdata:/var/lib/postgresql/data
      # Runs once on an empty volume -- creates the vector extension and schema.
      - ./migrations:/docker-entrypoint-initdb.d:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U gw -d gateway"]
      interval: 5s
      retries: 10

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      retries: 10

  api:
    build: .
    env_file: .env
    ports: ["8000:8000"]
    depends_on:
      postgres: {condition: service_healthy}
      redis: {condition: service_healthy}

  worker:
    build: .
    env_file: .env
    command: celery -A app.tasks.celery_app worker --loglevel=info --concurrency=4
    depends_on:
      postgres: {condition: service_healthy}
      redis: {condition: service_healthy}

  # Scheduler for the hourly cache reaper. Exactly ONE of these must run --
  # two beat processes double-schedule every periodic task.
  beat:
    build: .
    env_file: .env
    command: celery -A app.tasks.celery_app beat --loglevel=info
    depends_on:
      redis: {condition: service_healthy}

volumes:
  pgdata:
```

> **Does** — the entire stack: `postgres` (pgvector image) · `redis` · `api` · `worker` · `beat`.
> **Watch out** — `migrations/` is mounted into Postgres's init directory, so the schema runs **once,
> on an empty volume**. If you change the schema you need `docker compose down -v`, not just a
> restart — otherwise the old volume persists and your migration never runs.

---

## 31. `.devcontainer/devcontainer.json`

```json
{
  "name": "llm-gateway",
  "dockerComposeFile": "../docker-compose.yml",
  "service": "api",
  "workspaceFolder": "/srv",
  "features": {
    "ghcr.io/devcontainers/features/docker-in-docker:2": {}
  },
  "forwardPorts": [8000, 5432, 6379],
  "postCreateCommand": "pip install -e '.[dev]'",
  "customizations": {
    "vscode": {
      "extensions": ["ms-python.python", "charliermarsh.ruff"]
    }
  }
}
```

> **Does** — makes the repo one-click runnable in GitHub Codespaces.
> **Watch out** — this is what removes "you need a local machine with Docker" as a barrier. Someone
> reviewing your repo can open it in a browser and have the whole stack running.

> This is what makes the project runnable without a local machine: open the repo
> in **GitHub Codespaces** and everything — Postgres, Redis, the API, the worker —
> comes up in the browser.

## 32. `README.md`

This is what a reviewer reads first — most will read only this. Fill every `XX` with a measured
number; **never publish a figure you did not produce.**

````markdown
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
````

---
---

# PART III — RUN IT

## 33. Runbook

```bash
# 0. Setup -- Codespaces, or any machine with Docker
git clone https://github.com/<you>/llm-gateway && cd llm-gateway
cp .env.example .env

# 1. Everything up
docker compose up --build -d
curl -sf localhost:8000/healthz | python -m json.tool

# 2. Tests FIRST -- they need no provider key and catch paste errors in seconds
docker compose exec api pytest -q

# 3. Choose the threshold, then put it in .env and restart
docker compose exec api python bench/threshold_sweep.py
#    -> bench/results/threshold_curve.png, threshold.json

# 4. First request, then the same request again
curl -XPOST localhost:8000/v1/chat \
  -H "Authorization: Bearer demo-key-please-change" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'

# 5. Load
k6 run -e API_KEY=demo-key-please-change bench/load.js

run down one for k6 

MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" docker run --rm -i \
  --network llm-gateway_default \
  -v "$(pwd):/work" \
  -w /work \
  grafana/k6 run \
  -e BASE_URL=http://api:8000 \
  -e API_KEY=demo-key-please-change \
  bench/load.js

# 6. Cost curve
docker compose exec api python bench/cost_model.py


# 6b. Turn /metrics into the README tables (run AFTER k6)
docker compose exec api python bench/report.py

# 7. Chaos
bash bench/chaos.sh

# 7b. If you want CI later, this is the whole file:
#   .github/workflows/ci.yml -- ubuntu-latest, `services:` blocks for
#   pgvector/pgvector:pg16 and redis:7, then: pip install -e ".[dev]",
#   psql -f migrations/001_init.sql, pytest -q.
#   Worth it only if you will keep it green.

# 8. Tear down (the pgdata volume holds the cache; -v wipes it)
docker compose down          # keep data
docker compose down -v       # start clean
```

Everything above runs on one machine. There is no hosting step: the stack is five containers —
`postgres`, `redis`, `api`, `worker`, `beat` — and every number the README reports comes from it.

---

## 34. What each file does, and the order to do it

### 36.1 The dependency map

Nothing imports downward. `config.py` is imported by everything and imports nothing; `main.py`
imports everything and is imported by nothing. If you ever find yourself adding an import that
points *up* this list, something is in the wrong file.

**Layer 0 — constants and shapes. No imports from the project.**

| File | Job |
|---|---|
| `app/config.py` | Every tunable value: paths, thresholds, prices, timeouts, breaker settings. One place to change anything |
| `app/models.py` | The HTTP contract — `ChatRequest` / `ChatResponse`, deliberately OpenAI-shaped so a client switches by changing a base URL |
| `app/telemetry.py` | Prometheus metric definitions. Histogram buckets chosen for *this* system: a hit is ~16 ms and a miss ~900 ms, and library defaults put both in one bin |

**Layer 1 — infrastructure clients.**

| File | Job |
|---|---|
| `app/db.py` | The asyncpg pool, plus `register_vector` so a numpy array binds to `vector(384)` instead of being stringified |
| `app/limits.py` | Redis, and the three Lua scripts: token bucket, budget reserve, budget adjust. Also the single-flight lock. **Everything here is atomic on purpose** |

**Layer 2 — the provider seam.**

| File | Job |
|---|---|
| `app/providers/base.py` | `Provider` Protocol, `Usage`, `Completion`, `ProviderError`, and `cost_usd()` in `Decimal`. Everything downstream is provider-agnostic because of this file |
| `app/providers/mock.py` | Deterministic provider with a lognormal latency distribution and a failure dial. Every load and chaos test runs against it |
| `app/providers/groq.py` | Real provider. Primary |
| `app/providers/gemini.py` | Real provider. Failover target — a *different vendor*, so incidents are uncorrelated |

**Layer 3 — the actual logic.**

| File | Job |
|---|---|
| `app/cache/embed.py` | In-process sentence embedding. Loaded eagerly at startup so the cost lands in boot, not in someone's p99 |
| `app/cache/semantic.py` | The heart: `namespace()`, `is_eligible()`, **`lexical_conflict()`**, `lookup()`, `store()`. Guards 1–3 from §3 all live here |
| `app/breaker.py` | Circuit breaker with state in Redis, plus `call_with_failover()` walking the provider chain |
| `app/accounting.py` | Pessimistic cost `estimate()` for the reservation, and the idempotent ledger `record()` |
| `app/auth.py` | API key → tenant. SHA-256, not bcrypt — deliberately, and §24 says why |

**Layer 4 — orchestration.**

| File | Job |
|---|---|
| `app/tasks.py` | Celery. Cache persist, usage record, and the hourly TTL reaper. Everything that must not block the caller |
| `app/main.py` | The request path. Read it top to bottom and the whole design is visible |

**Not application code:**

| File | Job |
|---|---|
| `migrations/001_init.sql` | Schema. Runs once on an empty Postgres volume |
| `scripts/scaffold.py` | Writes all 37 files out of this document |
| `bench/threshold_sweep.py` | **The headline experiment** — produces the finding in §3 |
| `bench/cost_model.py` | Cost per 1,000 requests against hit rate |
| `bench/load.js` | k6. Custom metrics split hit and miss latency into separate distributions |
| `bench/report.py` | Scrapes `/metrics`, computes percentiles from histogram buckets, prints the README tables |
| `bench/chaos.sh` | Kills the primary provider under load |
| `tests/*` | 42 tests. `test_cache_correctness.py` is the one that matters |

---

### 36.2 The sequence, start to end

You are not typing 37 files. `scaffold.py` writes them all at once — so the sequence below is the
order in which you **bring the system up and prove each layer**, and each step has a stop condition.
Do not move on until it is met.

**Step 1 — get the tree on disk.**

```bash
mkdir -p llm-gateway/scripts && cd llm-gateway && git init
# paste DESIGN.md and scripts/scaffold.py, then:
python scripts/scaffold.py
```
*Stop condition:* `37 file blocks | 37 written`, no PROBLEMS.

**Step 2 — correctness before infrastructure.**

```bash
pip install -e ".[dev]"
pytest tests/test_cache_correctness.py -q
```
*Stop condition:* **25 passed.** These need no database and no Redis — only the embedding model — so
they run before anything else exists. If they fail, the paste is broken and nothing downstream is
worth debugging.

**Step 3 — bring up the stack.**

```bash
cp .env.example .env
docker compose up --build -d
curl -s localhost:8000/healthz
```
*Stop condition:* `{"ok": true, "providers": ["mock"]}`. If it hangs ~30 s first, that is the model
loading — expected once, at boot.

**Step 4 — the full suite, now that Redis exists.**

```bash
docker compose exec api pytest -q
```
*Stop condition:* **42 passed.** This is where `test_limits.py` proves the Lua rate limiter is
atomic and `test_breaker.py` proves failover works.

**Step 5 — prove the cache end to end.**

```bash
curl -XPOST localhost:8000/v1/chat -H "Authorization: Bearer demo-key-please-change" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}]}'
# wait a second for the Celery write, then send a PARAPHRASE:
curl -XPOST localhost:8000/v1/chat -H "Authorization: Bearer demo-key-please-change" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital city of France?"}]}'
```
*Stop condition:* the second response has `"cached": true` and `"similarity": ~0.957`. **This is the
moment the project works.** Then send the Germany variant and confirm it is `"cached": false` —
that is §3 happening in front of you.

**Step 6 — the experiments.**

```bash
docker compose exec api python bench/threshold_sweep.py     # the finding
docker compose exec api python bench/cost_model.py          # the cost curve
```
*Stop condition:* `threshold_curve.png` and `cost_curve.png` exist, and the sweep prints
**"FINDING: the distributions OVERLAP."**

**Step 7 — load.**

```bash
k6 run -e API_KEY=demo-key-please-change bench/load.js
docker compose exec api python bench/report.py
```
*Stop condition:* k6 reports `cache_hit_rate` and both latency trends; `report.py` prints the stage
table. Copy both into the README.

**Step 8 — chaos.**

```bash
bash bench/chaos.sh
```
*Stop condition:* `http_req_failed` under 1% while the primary is dead.

**Step 9 — fill in the README.** Every `XX` gets a number from steps 6–8. A README full of
placeholders reads as abandoned.

**Step 10 — confirm the doc and the code agree.**

```bash
python scripts/scaffold.py --check     # expect: 0 would change
git add -A && git commit -m "..." && git push
```

---

### 36.3 If you would rather build it in stages

Scaffolding everything and debugging upward is fastest. If you want to understand each layer as you
go instead, delete the files you have not reached yet and work in this order — it is the dependency
map, bottom up:

| Stage | Files | Proves |
|---|---|---|
| **1** | `config` · `models` · `telemetry` · `db` · `providers/*` · compose · Dockerfile | `/healthz` responds; k6 gives a **baseline with no cache** |
| **2** | `cache/embed` · `cache/semantic` · `tests/test_cache_correctness` · `bench/threshold_sweep` | The finding reproduces. **This is the project** |
| **3** | `limits` · `breaker` · `auth` · `tests/test_limits` · `tests/test_breaker` · `bench/chaos.sh` | Concurrency and failover hold |
| **4** | `accounting` · `tasks` · `main` · `bench/cost_model` · `bench/report` · README | Cost curve exists; every `XX` is measured |

Stage 1 exists to give you a **before** number. Without it you cannot claim the cache improved
anything — only assert it.

---

## 35. The four experiments

These *are* the project. The code is how you get them.

### 37.1 Threshold sweep — usefulness vs correctness

`python bench/threshold_sweep.py`. Sweeps cosine threshold over the labelled pairs, plots useful hit
rate against wrong-hit rate, picks the **lowest threshold with zero wrong hits**.

Report the **margin** — the gap between the worst paraphrase and the best adversarial pair. If it is
negative the classes overlap and no threshold works; the fix is a better embedding model, not more
tuning. Saying that unprompted is a strong signal.

### 37.2 Latency breakdown

From `gw_stage_latency_seconds`. Fill the table in the README. The number that matters is
**total-before-provider**: if it is not ~16 ms, something on the hot path should not be there.

### 37.3 Cost curve

`python bench/cost_model.py`. Cost per 1,000 requests against hit rate, with break-even marked.
Quote the monthly figure at your measured hit rate — that is the sentence that justifies the project.

### 37.4 Chaos

`bash bench/chaos.sh`. Kill the primary provider mid-load. Success is `http_req_failed` staying under
1% while `gw_provider_calls_total{provider="..."}` shows traffic shifting to the fallback.

---

## 36. Failure modes

| Symptom | Likely cause | Where to look |
|---|---|---|
| Cache returns a wrong answer | Guard bypassed, or `lookup()` called without the query text | §19 — `lexical_conflict` needs the *original prompt*, not just the vector |
| A negated or reordered prompt hits | Antonym list missing that pair | §19 `_ANTONYMS`; add it and add a regression pair to the test |
| Hit rate ~0% | Threshold too high, or every request lands in its own namespace | Check `temperature` is bucketed and the system prompt is stable |
| Hit rate suspiciously high | Namespace collapsed — e.g. system prompt not in the key | §19 `namespace()` |
| Rate limit passes ~2× the configured rate | Read-modify-write reintroduced | §20 — it must be one `evalsha` |
| Latency spikes under concurrency, CPU low | `embed.encode` called directly on the event loop | §26 — it must be `run_in_threadpool` |
| Provider stays open after recovery | Success path never calls `record_success` | §21 |
| A 400 takes the provider down | Non-retryable error counted as a breaker failure | §21 — check `e.retryable` |
| Budget drifts from the ledger | `settle_budget` skipped on an error path | §26 — every exit refunds or settles |
| `cache_entries` insert fails with a dimension error | Embedding model changed without a migration | §10 — `vector(384)` is fixed; changing models means a new column and a rebuild |
| Cold start ~3 s | Model downloading at runtime | §29 — it must be baked into the image |
| `pip install -e .` fails, "multiple top-level packages" | `[tool.setuptools] packages` missing | §7 — setuptools will not guess in a flat layout |
| Container restart-loops with no error message | OOM during model load | Give the `api` container ≥1 GB, or switch to `fastembed` (§4.1) |
| Celery retries double-count spend | `ON CONFLICT DO NOTHING` removed | §22 — `request_id` is UNIQUE for exactly this reason |
| N identical cold requests all call the provider | Single-flight lock not acquired, or `eligible` is False | §26 — non-cacheable requests deliberately skip coalescing |
| Waiters hang ~30 s after an upstream failure | `release_fill_lock` missing on the error path | §26 — every exit from the provider call must release |
| `cache_entries` grows without bound | Celery **beat** not running | §30 — `worker` and `beat` are separate processes |
| Periodic tasks fire twice | Two beat processes | Exactly one `beat` container, ever |

---
---

# PART IV — INTERVIEW PREP

## 37. Why these choices and not the alternatives

Every decision here has obvious alternatives and an interviewer will ask. Where an alternative is
genuinely defensible, this says so — claiming your choice is the only correct one is a worse answer
than acknowledging a real trade-off.

### 39.1 A gateway at all — why not just a library

| Alternative | Why not |
|---|---|
| **A shared SDK/wrapper library** | Every service must upgrade to get a fix, budgets cannot be enforced across services, and the cache is per-process so five replicas share nothing. Genuinely simpler for a single service |
| **Per-service caching** | Cache hit rate collapses: the same question asked by two services costs twice. Centralising is most of the win |
| **A managed product** (Portkey, Helicone, LiteLLM) | **Correct answer for a real company.** Rejected here because building it is the point, and because the cost/latency reasoning does not transfer if you did not do it |
| **✅ A gateway service** | One place for cost control, one cache, one failover policy, and clients change only a base URL |

### 39.2 Cache — why semantic and not exact-match

| Alternative | Why not |
|---|---|
| **Exact hash of the prompt** | Safe and useless: "what's the capital of France" and "What is the capital of France?" are different keys. Real traffic paraphrases constantly. **Ship this first** if you cannot measure a threshold |
| **Normalised hash** (lowercase, strip punctuation) | Catches trivial variation, misses everything semantic. Cheap; worth having *underneath* the semantic layer |
| **LLM-as-judge for equivalence** | Costs an LLM call to decide whether to skip an LLM call |
| **✅ Embedding similarity with structural guards** | Catches real paraphrase, and the wrong-answer risk is contained by namespacing, eligibility rules, a measured threshold and an adversarial test |

### 39.3 Cache correctness — why a lexical guard and not just a threshold

**This is the answer to prepare hardest**, because it is the one finding here that came out of
measurement rather than reading.

| Alternative | Why not |
|---|---|
| **Tune the threshold** | The standard advice, and it does not work: measured, an argument transposition scores **0.992** while a legitimate paraphrase scores **0.956**. The classes overlap, so no cut point separates them |
| **Use a better embedding model** | Would raise the margin and might separate them — genuinely the right long-term move, and worth saying. It does not remove the failure mode, because order-insensitivity is structural to bag-style sentence encoders |
| **Cross-encoder re-rank on candidate hits** | Much more accurate than bi-encoder cosine. Costs a model forward pass (~20–50 ms) inside the hot path, against a ~20 µs string check. Correct if hit rate matters more than latency |
| **LLM-as-judge on candidate hits** | An LLM call to decide whether to skip an LLM call |
| **Exact match only** | Safe, and gives up every real paraphrase |
| **✅ Cosine for recall + lexical guard for precision** | Uses each mechanism where it is strong: embeddings for topical closeness, string structure for the order and polarity that embeddings discard. Measured: wrong hits 2/10 → 0/10, useful hits unchanged at 6/6 |

**The sentence to lead with:** *"I assumed entity swaps would be the danger — France versus Germany.
Measured, that pair is 0.66 and perfectly safe. The real failures were negation at 0.965 and
argument transposition at 0.992, which sit above legitimate paraphrases at 0.956. So I stopped
trying to find a threshold and added a structural check for the thing embeddings throw away."*

### 39.4 Embedding — why in-process

| Alternative | Why not |
|---|---|
| **Hosted embedding API** (OpenAI, Cohere) | ~50 ms and real money to decide whether to skip a 900 ms call, plus a second dependency that can be down |
| **A separate embedding microservice** | A network hop for 5 ms of CPU. Justified only when you need GPU batching |
| **Cache the embedding in Redis** | Only helps on repeat *exact* strings — which the exact-match layer already caught |
| **✅ `all-MiniLM-L6-v2` in-process** | 80 MB, ~5 ms CPU, no network, no key, no extra failure domain — run in a threadpool so it does not block the loop |

### 39.5 Vector store — why pgvector

| Alternative | Why not |
|---|---|
| **Pinecone / Weaviate / Qdrant** | Better ANN at scale, but a second datastore, a second failure domain, and a network hop inside the latency budget. At ~10⁴ vectors per bucket the ANN advantage is not reachable |
| **FAISS in-process** | Fastest, but per-replica and non-durable: three replicas mean three caches and a restart empties them |
| **Redis vector search** | Plausible — Redis is already there. Rejected because the cache entries want durability and joins with the usage ledger |
| **✅ pgvector** | One datastore, transactional with the rest of the schema, exact KNN inside a filtered bucket is ~2 ms, and there is a clear scaling path (partition + per-partition HNSW) |

### 39.6 Rate limiting — why a token bucket in Lua

| Alternative | Why not |
|---|---|
| **Fixed window counter** | Allows 2× the limit across a window boundary — 100 requests at 0:59 and 100 more at 1:01 |
| **Sliding window log** | Accurate, but stores every request timestamp: O(n) memory per tenant |
| **Read-then-write in Python** | **Races.** Concurrent requests read the same count and all pass. This is the bug the concurrency test exists to catch |
| **A library** (`slowapi`, nginx `limit_req`) | Fine for per-IP limits; neither knows about per-tenant budgets or dynamic per-tenant rates from the database |
| **✅ Token bucket as a Lua script** | O(1) memory, allows bursts (which real clients have), and atomic by construction |

### 39.7 Budgets — why reserve then settle

| Alternative | Why not |
|---|---|
| **Charge after the call** | The cap is enforced *after* the money is spent, so a single expensive request can overshoot arbitrarily |
| **Charge a flat estimate, never reconcile** | Drifts from reality; the ledger and the counter disagree within a day |
| **Enforce in the database inside a transaction** | Correct, but a Postgres round-trip plus row lock on every request, and the row becomes a hot spot per tenant |
| **✅ Reserve pessimistically in Redis, settle with the actual** | The card-network authorise/capture pattern: the cap holds under the worst case, and the refund lands milliseconds later |

### 39.8 Async work — why Celery and not Kafka

| Alternative | Why not |
|---|---|
| **FastAPI `BackgroundTasks`** | Runs in the API process, so the work dies with the request if the pod is recycled, and it competes with request handling for CPU |
| **Kafka** | Right when you need replay, fan-out to multiple consumers, or an event log others subscribe to. Here there is one producer, one consumer, and no replay requirement — Kafka would be operational cost with no payoff |
| **`asyncio.create_task`** | Same fate-sharing problem as BackgroundTasks, plus no retry |
| **✅ Celery on Redis** | Retries with backoff, `acks_late` for redelivery if a worker dies, and Redis is already a dependency |

> Worth saying out loud: **if the usage ledger had to feed a data warehouse and a billing system and
> a real-time dashboard, this becomes Kafka.** The choice is about consumer count and replay, not
> about scale.

### 39.9 Circuit breaker — why Redis-backed

| Alternative | Why not |
|---|---|
| **In-process breaker** (`pybreaker`) | Each replica independently rediscovers the outage: N× the failed requests and N× disagreement about whether the provider is healthy |
| **Retry with backoff, no breaker** | Retries *add* load to a struggling provider. Backoff slows the stampede; it does not stop it |
| **Service mesh / Envoy** | Correct at scale, and free if you already run a mesh. Heavy for three containers |
| **✅ Shared state in Redis** | The first replica to notice protects the rest, and it is ~10 lines |

### 39.10 Load testing — why a mock provider

| Alternative | Why not |
|---|---|
| **Load test the real provider** | Measures *their* latency variance and *their* rate limits. You would publish someone else's p99 and pay for the privilege |
| **A local small model** (Ollama) | Real inference, but CPU-bound and unpredictable — it introduces the variance you are trying to control |
| **Record and replay real responses** | Good for correctness fixtures, wrong for load: playback has no latency distribution |
| **✅ Mock with an injected lognormal distribution** | Isolates gateway overhead, reproduces exactly, costs nothing, and lets failure rate be a *dial* for the chaos test |

### 39.12 Framing it as a system design answer

A system design round wants a different shape than "tell me about your project". Open with the
**constraint**, not the cache:

> "The dependency is 900 ms, costs money per token, is non-deterministic, and occasionally goes down.
> Every decision falls out of those four properties. So the first question is: what has to happen
> before I can answer the caller, and what can happen after?"

That establishes the sync/async split as *the* design and everything else hangs off it.

#### What this project lets you demonstrate

| Concept | Where it lives |
|---|---|
| **Sync/async split** | Cache write and cost accounting are off the hot path (§25) |
| **Cache key design** | Namespace = `(tenant, model, system_prompt, temperature_bucket)` (§19) |
| **Cache invalidation** | TTL at read time plus an hourly reaper — not pub/sub invalidation (§25) |
| **Cache stampede / single flight** | `SET NX` lock, waiters poll for the holder's result (§20, §26) |
| **Rate limiting** | Token bucket vs fixed window vs sliding log, and why Lua (§20) |
| **Atomicity under concurrency** | Read-then-write races; Redis executes Lua atomically (§20) |
| **Idempotency** | `usage_events.request_id` UNIQUE + `ON CONFLICT DO NOTHING` (§10, §22) |
| **Consistency models** | Budget counter in Redis is eventually consistent; the Postgres ledger is authoritative (§20) |
| **Stateless services** | Breaker state in Redis so replicas share one opinion (§21) |
| **Cascading-failure prevention** | Circuit breaker + cross-vendor failover (§21) |
| **Multi-tenancy** | Per-tenant buckets, budgets and cache namespaces — noisy-neighbour isolation |
| **Capacity reasoning** | CPU-bound hits vs I/O-bound misses (§4.1) |
| **Index selection** | Why *not* HNSW here, and what changes at 10⁵ rows per bucket (§10) |

#### Three deep dives they will probe

**"Walk me through the write path."**
> "There isn't one on the request path, deliberately. A cache write is a Postgres insert plus a
> 1.5 KB vector, done purely to benefit some *future* request — putting it in front of the caller
> trades their latency for someone else's. It goes to Celery. The cost is a wider stampede window,
> which is why there's a single-flight lock in front of the provider call."

**"How do you cap spend when you don't know the cost until after the call?"**
> "Authorise then capture, like a card network. Reserve a pessimistic estimate — assume the
> completion runs to `max_tokens` — then settle with the actual afterwards; a cache hit refunds the
> whole reservation. The counter is in Redis for speed and can drift by one in-flight request if a
> worker dies mid-call. The Postgres ledger is authoritative and reconciles it. That's a deliberate
> choice of availability over strict consistency on a counter, with an auditable record behind it."

**"What happens at 10× load?"**
> "Two different failures. Hit-heavy traffic saturates CPU on the embedding at ~800 req/s per
> instance — the service is stateless, so that's horizontal scaling and nothing else. Miss-heavy
> traffic saturates the *provider's* rate limit, which scaling cannot fix. That's where the real
> decision is: shed for free tenants, since a 429 is more honest than a 30-second wait, and queue
> for paying ones."

#### What to volunteer before they find it

Naming your own gaps is worth more than the feature would have been:

- **The antonym list is curated**, so it catches what I thought of. A WordNet-backed or learned
  check would generalise; a cross-encoder re-rank would be strictly better at 20–50 ms instead of
  20 µs.
- **Budgets are eventually consistent.** One in-flight request can escape the cap if a worker dies
  between the call and the settle.
- **Single region, no replication.** Designing multi-region without real traffic patterns would be
  guessing, and saying so is better than inventing a topology.
- **Load numbers come from a mock provider.** Deliberate — see §37.10 — but say it first.

### 39.11 Thirty-second answers

| Choice | The one sentence |
|---|---|
| A gateway | One place to enforce cost, one shared cache, one failover policy — clients change a base URL |
| Semantic cache | Real traffic paraphrases; exact-match misses all of it |
| The threshold | Measured — and measurement showed no threshold works alone, which is the actual finding |
| The lexical guard | Embeddings discard word order and polarity; "USD to EUR" and "EUR to USD" score 0.98 |
| Single flight | Without it, N identical cold requests all miss and all pay |
| The reaper | Read-time TTL hides expired rows; it does not delete them |
| Namespacing | Makes cross-tenant and cross-model hits impossible by construction rather than by tuning |
| Local embedding | The lookup must be cheaper than the call it avoids — 5 ms and free, versus 50 ms and billed |
| Threadpool | 5 ms of blocking CPU on the event loop serialises every concurrent request behind it |
| Lua rate limiter | Read-then-write races; Redis runs Lua atomically |
| Reserve/settle | You cannot know the cost before the call, so authorise pessimistically and capture the truth |
| Celery not Kafka | One producer, one consumer, no replay requirement |
| Redis breaker | Otherwise every replica rediscovers the same outage |
| Mock provider | Benchmarking a real provider measures their variance, not my gateway |

---

## Resume bullets

Every bullet below describes something that **exists in this repository** and produces a number you
can point at. Do not use one until you have run the thing it claims.

### The default five

Use these unless the role points elsewhere. They cover the finding, the concurrency work, the
failure handling and the methodology.

> **LLM Inference Gateway** · Python, FastAPI, Postgres/pgvector, Redis, Celery, Docker, k6
>
> - Built a **multi-tenant LLM gateway** with semantic caching, cutting cost per 1,000 requests from **\$X.XX to \$X.XX** at a **XX% hit rate** and p95 latency from **XXX ms to XX ms** on cache hits.
> - Measured a **wrong-answer failure mode** in semantic caching and showed *no similarity threshold can fix it*: argument-transposed prompts embed at **0.992** while legitimate paraphrases sit at **0.956**. Added a lexical guard on token order, polarity and negation, cutting wrong hits from **2/10 to 0/10** with useful hits unchanged.
> - Implemented **atomic per-tenant rate limiting and budget enforcement** as Redis Lua scripts using an authorise/settle pattern, with a concurrency test proving a burst-10 bucket admits exactly 10 of 50 simultaneous requests.
> - Added a **Redis-backed circuit breaker with cross-vendor failover**; a chaos test kills the primary provider under sustained k6 load with **<1% request failures**.
> - Load-tested against an **injected lognormal latency distribution** rather than a live provider, isolating gateway overhead at **~XX ms before the upstream call**.

---

### System-design bullet bank

Swap these in when the role is backend, platform, infrastructure or distributed systems. Each one is
tagged with what it signals, so you can pick against the job description rather than by feel.

**Request-path architecture**

> - Designed a **two-path request flow** answering in **~16 ms** on a cache hit by moving cache writes, cost accounting and usage rollups off the request path into Celery, so caller latency never pays for work that benefits future requests.

*Signals:* sync/async reasoning, latency budgeting. **This is the strongest single system-design
bullet in the project** — it is the answer to "what has to happen before you can respond".

**Concurrency correctness**

> - Eliminated a **cache-stampede** failure mode where N identical concurrent requests on a cold key each triggered a paid upstream call, using a Redis `SET NX` single-flight lock with a bounded 2 s wait and explicit release on the failure path.

*Signals:* thundering-herd awareness, and that you thought about the failure path — most people
forget the release.

**Idempotency / exactly-once**

> - Made the usage ledger **idempotent by database constraint** — `UNIQUE request_id` with `ON CONFLICT DO NOTHING` — so at-least-once Celery retries after a partial failure cannot double-count spend.

*Signals:* you know at-least-once delivery is the default and that idempotency belongs in the schema,
not in application hope.

**Capacity modelling**

> - Profiled the two request paths to **distinct bottlenecks** — cache hits CPU-bound on in-process embedding at **~800 req/s per 4-core instance**, cache misses I/O-bound and capped by upstream rate limits — and sized each independently rather than scaling one resource for both.

*Signals:* back-of-envelope capability. Interviewers grade this explicitly.

**Index selection**

> - Rejected an **HNSW vector index** after determining pgvector's post-filtering would silently return zero rows under a highly selective tenant predicate, using exact KNN inside a namespaced bucket instead and documenting the partitioning path beyond **10⁵ entries per bucket**.

*Signals:* you understand *why* an index behaves as it does, not just how to create one. The word
"silently" is doing the work — it is a correctness bug, not a performance one.

**Stateless service design**

> - Kept the service **stateless** by holding rate-limit buckets, circuit-breaker state and single-flight locks in Redis, so horizontal replicas share one view of dependency health instead of each independently rediscovering an outage.

*Signals:* you have thought past one process.

**Data lifecycle**

> - Prevented unbounded table growth from read-time TTL filtering with a **batched hourly reaper**, deleting in 5,000-row chunks so cleanup never holds locks that block inserts on the hot path.

*Signals:* operational maturity. The batching detail is the point — an unbounded `DELETE` is the
classic self-inflicted outage.

**Multi-tenancy**

> - Isolated tenants across **four dimensions** — rate limit, spend budget, cache namespace and usage ledger — so one tenant's burst or runaway loop cannot affect another's latency, cost, or cache correctness.

*Signals:* noisy-neighbour awareness. The cache-namespace dimension is the non-obvious one: without
it, similarity search can cross tenants.

**Observability**

> - Instrumented **per-stage latency histograms** with buckets tuned to the system's bimodal profile (~16 ms hits, ~900 ms misses) and wrote a reporter computing percentiles from cumulative bucket counters, so latency analysis needs no Prometheus server.

*Signals:* you know default histogram buckets are usually wrong, and that a bimodal distribution has
no meaningful average.

**Cost engineering**

> - Modelled **cost per 1,000 requests as a function of cache hit rate** with an explicit break-even against calling the provider directly, and enforced per-tenant monthly caps with a pessimistic reserve that settles to actual spend after each call.

*Signals:* infrastructure cost reasoning, now graded explicitly in system design rounds.

---

### Picking

- **Backend / distributed systems** → default #2 and #3, plus request-path architecture, concurrency
  correctness and idempotency.
- **Platform / infrastructure / SRE** → stateless design, data lifecycle, observability, capacity.
- **AI or ML platform** → default #1 and #2, plus cost engineering and capacity.
- **Generalist SDE** → the default five as written.

Two rules regardless of role:

1. **Never more than six bullets** for one project. Six is a wall; five is read.
2. **Lead with the finding** (default #2) if you only get one. It is the only bullet that says you
   measured something, were wrong, and changed approach — and that is rarer than any technology on
   the list.

Fill every `X` with a measured number. A bullet with a placeholder in it is worse than no bullet.
