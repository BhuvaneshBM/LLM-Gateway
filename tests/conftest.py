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
