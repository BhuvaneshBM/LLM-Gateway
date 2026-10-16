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
