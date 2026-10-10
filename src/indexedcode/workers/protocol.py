"""Object shapes for the versioned worker HTTP contract (plan §4.2–4.4)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from indexedcode.models import Backend

FINGERPRINT_PATTERN = r"^sha256:[0-9a-f]{64}$"


class ModelInfo(BaseModel):
    """GET /v1/model — exact model identity and configuration."""

    model: str
    revision: str
    backend: Backend
    fingerprint: str = Field(pattern=FINGERPRINT_PATTERN)
    dimension: int | None = Field(default=None, ge=1)
    pooling: str | None = None
    normalized: bool | None = None
    quantization: str | None = None


class EmbedResponse(BaseModel):
    """POST /v1/embed — one vector per input text."""

    vectors: list[list[float]] = Field(min_length=0)
    dimension: int = Field(ge=1)
    fingerprint: str = Field(pattern=FINGERPRINT_PATTERN)


class SummaryResponse(BaseModel):
    """POST /v1/summarize — deterministic decoding result."""

    summary: str
    fingerprint: str = Field(pattern=FINGERPRINT_PATTERN)
