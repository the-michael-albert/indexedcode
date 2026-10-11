"""Hosted embedding and summarization API clients (``mode = "api"``).

These expose the same ``model_info``/``embed``/``summarize`` surface as
:class:`~indexedcode.workers.client.WorkerClient`, so the semantic pipeline
treats a hosted API like any other worker. Two wire formats are supported:

* ``openai`` — the OpenAI-compatible ``/embeddings`` and
  ``/chat/completions`` endpoints (OpenAI, Voyage, Together, Fireworks,
  Ollama, vLLM, ...).
* ``anthropic`` — the Messages API (summarization only).

Stdlib only. 429 and 5xx responses are retried with exponential backoff,
honoring ``Retry-After``; the API key is read from an environment variable
and never written to configuration or logs.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

from indexedcode.workers.client import classify_status
from indexedcode.workers.errors import (
    WorkerAuthError,
    WorkerProtocolError,
    WorkerTransientError,
)
from indexedcode.workers.fingerprint import model_fingerprint
from indexedcode.workers.protocol import EmbedResponse, SummaryResponse

PROVIDERS = ("openai", "anthropic")
ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
}
DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
DEFAULT_MODELS = {
    ("openai", "embedding"): "text-embedding-3-small",
    ("openai", "summary"): "gpt-4o-mini",
    ("anthropic", "summary"): "claude-haiku-5-5",
}


def default_api_model(provider: str, role: str) -> str | None:
    """Default model for ``provider``/``role`` (``summary``/``embedding``)."""
    return DEFAULT_MODELS.get((provider, role))


class _ApiBase:
    def __init__(self, provider, base_url, api_key, model, timeout_s=60.0,
                 max_retries=5, sleep=time.sleep):
        self.provider = provider
        self.base_url = (base_url or DEFAULT_BASE_URLS[provider]).rstrip("/")
        self._api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._sleep = sleep
        self.expected_fingerprint = ""

    def _headers(self) -> dict:
        if self.provider == "anthropic":
            return {"x-api-key": self._api_key,
                    "anthropic-version": ANTHROPIC_VERSION}
        return {"Authorization": f"Bearer {self._api_key}"}

    def _post(self, path: str, payload: dict) -> dict:
        url = self.base_url + path
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(url, data=body, method="POST")
            request.add_header("Content-Type", "application/json")
            for key, value in self._headers().items():
                request.add_header(key, value)
            try:
                with urllib.request.urlopen(
                        request, timeout=self.timeout_s) as response:
                    return json.loads(response.read() or b"{}")
            except urllib.error.HTTPError as exc:
                code = exc.code
                if code in (401, 403):
                    raise WorkerAuthError(
                        f"POST {path}: {self.provider} rejected the API key"
                        f" (HTTP {code})") from exc
                if classify_status(code) == "fatal":
                    raise WorkerProtocolError(
                        f"POST {path}: {self.provider} error (HTTP {code}):"
                        f" {_error_detail(exc)}") from exc
                failure = WorkerTransientError(
                    f"POST {path}: transient {self.provider} failure"
                    f" (HTTP {code})")
                delay = _retry_after(exc)
                cause = exc
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                failure = WorkerTransientError(
                    f"POST {path}: {self.provider} unreachable ({exc})")
                delay, cause = None, exc
            if attempt >= self.max_retries:
                raise failure from cause
            self._sleep(delay if delay is not None
                        else min(2 ** attempt, 30))
        raise AssertionError("unreachable")  # pragma: no cover

    def model_info(self):
        """Identity comes from configuration; no network round trip."""
        return SimpleNamespace(
            model=self.model, fingerprint=self.expected_fingerprint)


class ApiEmbeddingClient(_ApiBase):
    """OpenAI-compatible embeddings endpoint."""

    def __init__(self, provider, base_url, api_key, model, dimensions=None,
                 **kwargs):
        if provider != "openai":
            raise ValueError(
                f"embeddings support the openai provider only, not"
                f" {provider!r}")
        super().__init__(provider, base_url, api_key, model, **kwargs)
        self.dimensions = dimensions
        details = {"provider": provider, "model": model}
        if dimensions:
            details["dimension"] = dimensions
        self.expected_fingerprint = model_fingerprint(**details)

    def embed(self, texts) -> EmbedResponse:
        texts = list(texts)
        payload = {"model": self.model, "input": texts}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        data = self._post("/embeddings", payload).get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise WorkerProtocolError(
                f"embeddings API returned {len(data or [])} vectors for"
                f" {len(texts)} texts")
        # the API tags each row with its input position; don't trust order
        rows = sorted(data, key=lambda row: row.get("index", 0))
        vectors = [_unit(row.get("embedding")) for row in rows]
        return EmbedResponse(
            vectors=vectors, dimension=len(vectors[0]),
            fingerprint=self.expected_fingerprint)


def _unit(vector):
    if not isinstance(vector, list) or not vector:
        raise WorkerProtocolError("embeddings API returned a bad vector")
    norm = math.sqrt(sum(v * v for v in vector))
    if not math.isfinite(norm) or norm == 0.0:
        raise WorkerProtocolError("embeddings API returned a zero vector")
    return [v / norm for v in vector]


class ApiSummaryClient(_ApiBase):
    """Chat-completion summarizer (OpenAI-compatible or Anthropic)."""

    def __init__(self, provider, base_url, api_key, model, **kwargs):
        super().__init__(provider, base_url, api_key, model, **kwargs)
        self.expected_fingerprint = model_fingerprint(
            provider=provider, model=model)

    def summarize(self, prompt: str, max_output_tokens: int) -> SummaryResponse:
        if self.provider == "anthropic":
            data = self._post("/messages", {
                "model": self.model,
                "max_tokens": max_output_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            })
            text = "".join(
                block.get("text", "") for block in data.get("content") or []
                if block.get("type") == "text")
        else:
            data = self._post("/chat/completions", {
                "model": self.model,
                "max_tokens": max_output_tokens,
                "temperature": 0,
                "messages": [{"role": "user", "content": prompt}],
            })
            choices = data.get("choices") or [{}]
            text = (choices[0].get("message") or {}).get("content") or ""
        return SummaryResponse(
            summary=text, fingerprint=self.expected_fingerprint)


def _retry_after(exc) -> float | None:
    value = exc.headers.get("Retry-After") if exc.headers else None
    try:
        return min(float(value), 60.0) if value is not None else None
    except ValueError:
        return None


def _error_detail(exc) -> str:
    try:
        return exc.read().decode("utf-8", "replace")[:300]
    except Exception:  # noqa: BLE001 - diagnostics only
        return ""
