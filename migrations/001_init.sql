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
