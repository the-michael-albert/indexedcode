"""Authenticated worker HTTP client (plan §4.2–4.4).

One client contract serves both managed loopback workers and remote
endpoints. Uses only the stdlib. Retry classification: transport errors,
5xx, and 429 are transient; 401/403 are auth failures; other 4xx are
fatal protocol violations.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from indexedcode.workers.errors import (
    FingerprintMismatchError,
    WorkerAuthError,
    WorkerProtocolError,
    WorkerTransientError,
)
from indexedcode.workers.protocol import EmbedResponse, ModelInfo, SummaryResponse


def classify_status(status_code: int) -> str:
    """Classify an HTTP status as "transient" (retryable) or "fatal"."""
    if status_code >= 500 or status_code == 429:
        return "transient"
    return "fatal"


def is_retryable(status_code: int) -> bool:
    return classify_status(status_code) == "transient"


class WorkerClient:
    """Bearer-authenticated client for the versioned worker contract."""

    def __init__(self, base_url: str, secret: str, expected_fingerprint: str,
                 timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.secret = secret
        self.expected_fingerprint = expected_fingerprint
        self.timeout_s = timeout_s

    def _request(self, method: str, path: str, payload=None) -> dict:
        url = self.base_url + path
        data = None
        headers = {"Authorization": f"Bearer {self.secret}"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(url, data=data, method=method)
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout_s
            ) as response:
                body = response.read() or b"{}"
        except urllib.error.HTTPError as exc:
            code = exc.code
            if code in (401, 403):
                raise WorkerAuthError(
                    f"{method} {path}: worker rejected credentials"
                    f" (HTTP {code})"
                ) from exc
            if is_retryable(code):
                raise WorkerTransientError(
                    f"{method} {path}: transient worker failure (HTTP {code})"
                ) from exc
            raise WorkerProtocolError(
                f"{method} {path}: fatal worker error (HTTP {code})"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise WorkerTransientError(
                f"{method} {path}: worker unreachable ({exc})"
            ) from exc
        return json.loads(body)

    def _check_fingerprint(self, fingerprint: str) -> None:
        if fingerprint != self.expected_fingerprint:
            raise FingerprintMismatchError(
                f"worker fingerprint {fingerprint!r} does not match expected"
                f" {self.expected_fingerprint!r}"
            )

    def model_info(self) -> ModelInfo:
        info = ModelInfo(**self._request("GET", "/v1/model"))
        self._check_fingerprint(info.fingerprint)
        return info

    def embed(self, texts) -> EmbedResponse:
        response = EmbedResponse(
            **self._request("POST", "/v1/embed", {"texts": list(texts)})
        )
        self._check_fingerprint(response.fingerprint)
        return response

    def summarize(self, prompt: str, max_output_tokens: int) -> SummaryResponse:
        response = SummaryResponse(**self._request("POST", "/v1/summarize", {
            "prompt": prompt,
            "max_output_tokens": max_output_tokens,
        }))
        self._check_fingerprint(response.fingerprint)
        return response

    def status(self) -> dict:
        return self._request("GET", "/v1/status")
