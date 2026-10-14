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
