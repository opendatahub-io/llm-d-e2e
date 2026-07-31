"""OpenAI-compatible HTTP client for vLLM inference endpoints."""

from __future__ import annotations

import httpx


class LLMClient:
    """Client for health checks, model listing, and inference against vLLM."""

    def __init__(self, base_url: str, bearer_token: str = "", timeout: float = 120):
        headers = {}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"
        self._client = httpx.Client(
            base_url=base_url,
            headers=headers,
            verify=False,
            timeout=timeout,
        )

    def close(self):
        self._client.close()

    def health_check(self) -> bool:
        r = self._client.get("/health")
        r.raise_for_status()
        return True

    def list_models(self) -> dict:
        r = self._client.get("/v1/models")
        r.raise_for_status()
        return r.json()

    def completions(self, model: str, prompt: str, max_tokens: int = 64, temperature: float = 0.1) -> dict:
        r = self._client.post(
            "/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        r.raise_for_status()
        return r.json()

    def chat(self, model: str, prompt: str | list[dict], max_tokens: int = 64, temperature: float = 0.1) -> dict:
        messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        r = self._client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        r.raise_for_status()
        return r.json()
