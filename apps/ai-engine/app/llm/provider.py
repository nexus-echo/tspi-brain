"""LLM provider abstraction.

The LLM only *writes* the report from grounded facts; it does not store knowledge.
Supported: DeepSeek (OpenAI-compatible chat API) and Ollama (local). Selection follows
`LLM_PRIMARY`, then falls back to any other provider that has credentials — so if DeepSeek
is configured it is used even when Ollama isn't running. Disabled in tests.
"""
from __future__ import annotations

import httpx

from app.config import settings


class LLMProvider:
    def __init__(self) -> None:
        # Build an ordered list of providers to try (primary first, then any with creds).
        order: list[str] = []
        if settings.llm_primary:
            order.append(settings.llm_primary)
        if settings.openrouter_api_key and "openrouter" not in order:
            order.append("openrouter")
        if settings.deepseek_api_key and "deepseek" not in order:
            order.append("deepseek")
        if settings.ollama_base_url and "ollama" not in order:
            order.append("ollama")
        # keep only providers that are actually usable
        self.chain = [p for p in order if self._usable(p)]
        self.enabled = bool(self.chain) and settings.tspi_env != "test"

    @staticmethod
    def _usable(provider: str) -> bool:
        if provider == "openrouter":
            return bool(settings.openrouter_api_key)
        if provider == "deepseek":
            return bool(settings.deepseek_api_key)
        if provider == "ollama":
            return bool(settings.ollama_base_url)
        return False

    async def complete(self, prompt: str) -> str:
        """Try each provider in the chain until one returns text."""
        last_err: Exception | None = None
        for provider in self.chain:
            try:
                if provider == "openrouter":
                    return await self._openrouter(prompt)
                if provider == "deepseek":
                    return await self._deepseek(prompt)
                if provider == "ollama":
                    return await self._ollama(prompt)
            except Exception as e:  # noqa: BLE001 — try the next provider
                last_err = e
        if last_err:
            raise last_err
        raise RuntimeError("No LLM provider available.")

    async def _openrouter(self, prompt: str) -> str:
        url = f"{settings.openrouter_base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}
        payload = {
            "model": settings.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _deepseek(self, prompt: str) -> str:
        url = f"{settings.deepseek_base_url.rstrip('/')}/chat/completions"
        headers = {"Authorization": f"Bearer {settings.deepseek_api_key}"}
        payload = {
            "model": settings.deepseek_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def _ollama(self, prompt: str) -> str:
        url = f"{settings.ollama_base_url.rstrip('/')}/api/generate"
        payload = {"model": settings.ollama_model, "prompt": prompt, "stream": False}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")


class OllamaLocal:
    """Direct local Ollama client for extraction (vision + strict-JSON text).

    Kept separate from the report-writing chain: extraction is LOCAL-ONLY by governance
    decision, so it never falls back to a cloud provider. All calls degrade gracefully.
    """

    def __init__(self) -> None:
        self.base = settings.ollama_base_url.rstrip("/")

    async def vision_json(self, prompt: str, images_b64: list[str], model: str | None = None) -> str:
        """Ask a local vision model to read image(s) and return strict JSON text."""
        payload = {
            "model": model or settings.vision_model,
            "prompt": prompt,
            "images": images_b64,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        async with httpx.AsyncClient(timeout=settings.extraction_timeout_s) as client:
            resp = await client.post(f"{self.base}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")

    async def text_json(self, prompt: str, model: str | None = None) -> str:
        """Ask a local text model to structure text and return strict JSON text."""
        payload = {
            "model": model or (settings.extraction_text_model or settings.ollama_model),
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        async with httpx.AsyncClient(timeout=settings.extraction_timeout_s) as client:
            resp = await client.post(f"{self.base}/api/generate", json=payload)
            resp.raise_for_status()
            return resp.json().get("response", "")
