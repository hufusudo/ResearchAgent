"""Embedding abstraction supporting local deterministic hashing and OpenAI-compatible APIs."""

from __future__ import annotations

import hashlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol, runtime_checkable

import httpx
import numpy as np

from core.config import Settings


@runtime_checkable
class EmbedderProtocol(Protocol):
    """Protocol for dense text embedding providers."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        ...


class LocalHashEmbedder:
    """Zero-dependency deterministic feature-hashing embedder for fast offline testing."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._stats_lock = threading.Lock()
        self._usage_stats: dict[str, Any] = {
            "calls": 0,
            "texts_embedded": 0,
            "elapsed_seconds": 0.0,
        }

    def get_usage_stats(self) -> dict[str, Any]:
        """Return cumulative embedding call statistics."""
        with self._stats_lock:
            return {
                "calls": int(self._usage_stats["calls"]),
                "texts_embedded": int(self._usage_stats["texts_embedded"]),
                "elapsed_seconds": round(float(self._usage_stats["elapsed_seconds"]), 3),
            }

    def reset_usage_stats(self) -> None:
        """Reset cumulative embedding usage counters."""
        with self._stats_lock:
            self._usage_stats = {"calls": 0, "texts_embedded": 0, "elapsed_seconds": 0.0}

    def _hash_feature(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "little", signed=False)
        index = value % self.dim
        sign = 1.0 if ((value >> 63) & 1) == 0 else -1.0
        return index, sign

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            vec[0] = 1.0
            return vec.tolist()

        for tok in tokens:
            idx, sign = self._hash_feature(f"w:{tok}")
            vec[idx] += 1.5 * sign
            if len(tok) >= 3:
                for i in range(len(tok) - 2):
                    trigram = tok[i : i + 3]
                    c_idx, c_sign = self._hash_feature(f"c:{trigram}")
                    vec[c_idx] += 0.5 * c_sign

        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        else:
            vec[0] = 1.0
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        t0 = time.perf_counter()
        vectors = [self._embed_one(text) for text in texts]
        elapsed = time.perf_counter() - t0
        with self._stats_lock:
            self._usage_stats["calls"] += 1
            self._usage_stats["texts_embedded"] += len(texts)
            self._usage_stats["elapsed_seconds"] += elapsed
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbedder:
    """OpenAI-compatible /embeddings HTTP client with concurrent batching and retries."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        batch_size: int = 16,
        timeout: float = 60.0,
        max_workers: int = 1,
        max_retries: int = 4,
        base_delay: float = 1.5,
    ) -> None:
        if not api_key:
            raise ValueError(
                "An API key is required for EMBEDDING_PROVIDER=openai. "
                "Set EMBEDDING_API_KEY or LLM_API_KEY in .env."
            )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.max_workers = max(1, max_workers)
        self.max_retries = max(1, max_retries)
        self.base_delay = base_delay
        self._stats_lock = threading.Lock()
        self._usage_stats: dict[str, Any] = {
            "calls": 0,
            "texts_embedded": 0,
            "elapsed_seconds": 0.0,
        }

    def get_usage_stats(self) -> dict[str, Any]:
        """Return thread-safe cumulative embedding call statistics."""
        with self._stats_lock:
            return {
                "calls": int(self._usage_stats["calls"]),
                "texts_embedded": int(self._usage_stats["texts_embedded"]),
                "elapsed_seconds": round(float(self._usage_stats["elapsed_seconds"]), 3),
            }

    def reset_usage_stats(self) -> None:
        """Reset cumulative embedding usage counters."""
        with self._stats_lock:
            self._usage_stats = {"calls": 0, "texts_embedded": 0, "elapsed_seconds": 0.0}

    def _embed_batch(
        self,
        endpoint: str,
        headers: dict[str, str],
        batch: list[str],
    ) -> list[list[float]]:
        t0 = time.perf_counter()
        last_error: Exception | None = None
        retries = max(self.max_retries, 4)
        for attempt in range(retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    response = client.post(
                        endpoint,
                        headers=headers,
                        json={"model": self.model, "input": batch},
                    )
                if response.status_code in {429, 500, 502, 503, 504}:
                    last_error = RuntimeError(
                        f"Embedding API failed ({response.status_code}): {response.text[:300]}"
                    )
                    if attempt < retries - 1:
                        time.sleep(self.base_delay * (2**attempt))
                        continue
                    break
                if response.status_code >= 400:
                    raise RuntimeError(
                        f"Embedding API failed ({response.status_code}): {response.text[:300]}"
                    )
                payload = response.json()
                data = sorted(payload["data"], key=lambda item: item["index"])
                vectors = [item["embedding"] for item in data]
                elapsed = time.perf_counter() - t0
                with self._stats_lock:
                    self._usage_stats["calls"] += 1
                    self._usage_stats["texts_embedded"] += len(batch)
                    self._usage_stats["elapsed_seconds"] += elapsed
                return vectors
            except (httpx.HTTPError, OSError) as exc:
                last_error = exc
                if attempt < retries - 1:
                    time.sleep(self.base_delay * (2**attempt))
                    continue
                break

        if len(batch) > 1:
            mid = len(batch) // 2
            left = self._embed_batch(endpoint, headers, batch[:mid])
            right = self._embed_batch(endpoint, headers, batch[mid:])
            return left + right

        if isinstance(last_error, httpx.HTTPError | OSError):
            raise RuntimeError(f"Embedding API request failed: {last_error}") from last_error
        raise RuntimeError(f"Embedding API failed after retries: {last_error}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Connection": "close",
        }
        endpoint = f"{self.base_url}/embeddings"
        batches = [
            [(t[:4000] if t.strip() else " ") for t in texts[start : start + self.batch_size]]
            for start in range(0, len(texts), self.batch_size)
        ]

        if len(batches) <= 1 or self.max_workers <= 1:
            embeddings: list[list[float]] = []
            for batch in batches:
                embeddings.extend(self._embed_batch(endpoint, headers, batch))
            return embeddings

        with ThreadPoolExecutor(max_workers=min(self.max_workers, len(batches))) as executor:
            futures = [
                executor.submit(self._embed_batch, endpoint, headers, batch)
                for batch in batches
            ]
            ordered_embeddings: list[list[float]] = []
            for fut in futures:
                ordered_embeddings.extend(fut.result())
            return ordered_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_embedder(settings: Settings, force_local: bool = False) -> EmbedderProtocol:
    """Create the configured embedder (or local hash embedder when forced)."""
    if force_local or settings.embedding_provider == "local_hash":
        return LocalHashEmbedder()
    return OpenAIEmbedder(
        api_key=settings.effective_embedding_api_key,
        base_url=settings.effective_embedding_base_url,
        model=settings.embedding_model,
    )
