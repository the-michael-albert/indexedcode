"""Embedding serialization and validation tests: little-endian float32,
finite values, nonzero norm, normalization, dimension match (plan §5.2)."""

from __future__ import annotations

import math
import struct

import pytest

from indexedcode.embeddings import EmbeddingValidationError, pack, validate


def test_pack_is_little_endian_float32():
    blob = pack([1.0, -0.5])
    assert blob == struct.pack("<2f", 1.0, -0.5)
    assert len(blob) == 8


def test_validate_accepts_normalized_finite_vector():
    vec = [0.5, 0.5, 0.5, 0.5]
    assert validate(vec, dimensions=4) == vec


def test_validate_rejects_wrong_dimension():
    with pytest.raises(EmbeddingValidationError):
        validate([0.5, 0.5], dimensions=4)


def test_validate_rejects_nonfinite_values():
    with pytest.raises(EmbeddingValidationError):
        validate([float("nan"), 0.5, 0.5, 0.5], dimensions=4)
    with pytest.raises(EmbeddingValidationError):
        validate([float("inf"), 0.5, 0.5, 0.5], dimensions=4)


def test_validate_rejects_zero_vector():
    with pytest.raises(EmbeddingValidationError):
        validate([0.0, 0.0, 0.0, 0.0], dimensions=4)


def test_validate_rejects_unnormalized_vector():
    # norm = 2.0, not ~1.0
    with pytest.raises(EmbeddingValidationError):
        validate([1.0, 1.0, 1.0, 1.0], dimensions=4)


def test_validate_tolerance():
    nearly = 1.0 / math.sqrt(3)
    assert validate([nearly, nearly, nearly], dimensions=3, tolerance=1e-3)
