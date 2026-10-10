"""Embedding serialization and validation (plan §5.2): little-endian
float32 blobs of finite, nonzero, normalized vectors."""

from __future__ import annotations

import math
import struct


class EmbeddingValidationError(ValueError):
    """A vector failed dimension, finiteness, or normalization checks."""


def pack(values):
    """Serialize floats to a little-endian float32 blob."""
    return struct.pack("<" + "f" * len(values), *values)


def validate(vector, dimensions, tolerance=1e-6):
    """Verify dimension count, finiteness, nonzero norm, and unit norm.

    Returns the vector unchanged on success; raises otherwise.
    """
    if len(vector) != dimensions:
        raise EmbeddingValidationError(
            f"expected {dimensions} dimensions, got {len(vector)}"
        )
    for value in vector:
        if not math.isfinite(value):
            raise EmbeddingValidationError(
                f"nonfinite embedding value: {value!r}"
            )
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0.0:
        raise EmbeddingValidationError("embedding vector has zero norm")
    if abs(norm - 1.0) > tolerance:
        raise EmbeddingValidationError(
            f"embedding vector is not normalized (norm={norm:.6f})"
        )
    return vector
