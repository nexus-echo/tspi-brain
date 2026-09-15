"""Embedding provider — one interface, three backends, sync + async.

- 'hash'   (default): dependency-free hashed vector. Dev/CI; no network.
- 'api'    : OpenAI-compatible embeddings API (managed, high quality).
- 'ollama' : LOCAL Ollama embedding model (private, no per-token cost).

Sync methods (embed / embed_many) are used by the offline embed_knowledge script.
Async methods (aembed / aembed_many) are used at query time so the event loop never blocks.
"""
from __future__ import annotations

import hashlib
import math
import re

from app.config import settings

_TOKEN = re.compile(r"[a-z0-9]+")


class EmbeddingProvider:
    def __init__(self) -> None:
        self.backend = getattr(settings, "embedding_backend", "hash")
        self.dim = int(getattr(settings, "embedding_dim", 384))
        if self.backend == "api" and not settings.embedding_api_key:
            raise RuntimeError("embedding_backend='api' but EMBEDDING_API_KEY is not set.")
        if self.backend == "ollama" and not settings.ollama_base_url:
            raise RuntimeError("embedding_backend='ollama' but OLLAMA_BASE_URL is not set.")

    @property
    def model_id(self) -> str:
        if self.backend == "api":
            return settings.embedding_model
        if self.backend == "ollama":
            return settings.ollama_embedding_model
        return "hash"

    # --- sync (offline batch) ---
    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "hash":
            return [self._hash_embed(t) for t in texts]
        if self.backend == "api":
            return self._api_embed(texts)
        if self.backend == "ollama":
            return self._ollama_embed(texts)
        raise NotImplementedError(f"embedding backend '{self.backend}' not supported")

    # --- async (query time) ---
    async def aembed(self, text: str) -> list[float]:
        return (await self.aembed_many([text]))[0]

    async def aembed_many(self, texts: list[str]) -> list[list[float]]:
        if self.backend == "hash":
            return [self._hash_embed(t) for t in texts]
        if self.backend == "api":
            return await self._api_embed_async(texts)
        if self.backend == "ollama":
            return await self._ollama_embed_async(texts)
        raise NotImplementedError(f"embedding backend '{self.backend}' not supported")

    # --- hash backend ---
    def _hash_embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in _TOKEN.findall((text or "").lower()):
            feats = [tok] + [tok[i:i + 3] for i in range(max(0, len(tok) - 2))]
            for feat in feats:
                h = int(hashlib.md5(feat.encode()).hexdigest(), 16)
                vec[h % self.dim] += 1.0 if (h >> 7) & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _check_dim(self, vecs: list[list[float]]) -> list[list[float]]:
        if vecs and len(vecs[0]) != self.dim:
            raise RuntimeError(
                f"EMBEDDING_DIM={self.dim} but model '{self.model_id}' returned "
                f"{len(vecs[0])}-d vectors. Set EMBEDDING_DIM={len(vecs[0])}.")
        return vecs

    # --- api backend ---
    def _api_payload(self, texts: list[str]) -> tuple[str, dict, dict]:
        url = f"{settings.embedding_api_base.rstrip('/')}/embeddings"
        headers = {"Authorization": f"Bearer {settings.embedding_api_key}"}
        payload: dict = {"model": settings.embedding_model, "input": texts}
        if self.dim and settings.embedding_api_send_dimensions:
            payload["dimensions"] = self.dim
        return url, payload, headers

    def _api_embed(self, texts: list[str]) -> list[list[float]]:
        import httpx
        url, payload, headers = self._api_payload(texts)
        with httpx.Client(timeout=60) as c:
            r = c.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data]

    async def _api_embed_async(self, texts: list[str]) -> list[list[float]]:
        import httpx
        url, payload, headers = self._api_payload(texts)
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(url, json=payload, headers=headers)
            r.raise_for_status()
            data = sorted(r.json()["data"], key=lambda d: d["index"])
            return [d["embedding"] for d in data]

    # --- ollama backend ---
    def _ollama_embed(self, texts: list[str]) -> list[list[float]]:
        import httpx
        base = settings.ollama_base_url.rstrip("/")
        model = settings.ollama_embedding_model
        with httpx.Client(timeout=120) as c:
            r = c.post(f"{base}/api/embed", json={"model": model, "input": texts})
            if r.status_code == 200 and "embeddings" in r.json():
                vecs = r.json()["embeddings"]
            else:
                vecs = []
                for t in texts:
                    rr = c.post(f"{base}/api/embeddings", json={"model": model, "prompt": t})
                    rr.raise_for_status()
                    vecs.append(rr.json()["embedding"])
        return self._check_dim(vecs)

    async def _ollama_embed_async(self, texts: list[str]) -> list[list[float]]:
        import httpx
        base = settings.ollama_base_url.rstrip("/")
        model = settings.ollama_embedding_model
        async with httpx.AsyncClient(timeout=120) as c:
            r = await c.post(f"{base}/api/embed", json={"model": model, "input": texts})
            if r.status_code == 200 and "embeddings" in r.json():
                vecs = r.json()["embeddings"]
            else:
                vecs = []
                for t in texts:
                    rr = await c.post(f"{base}/api/embeddings", json={"model": model, "prompt": t})
                    rr.raise_for_status()
                    vecs.append(rr.json()["embedding"])
        return self._check_dim(vecs)
