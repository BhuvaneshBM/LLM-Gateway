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
