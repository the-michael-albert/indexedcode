"""Worker protocol object shapes (plan §4.2–4.4)."""

import pytest

from indexedcode.workers.protocol import (
    EmbedResponse,
    ModelInfo,
    SummaryResponse,
)

VALID_MODEL = {
    "model": "fake-embed",
    "revision": "r1",
    "backend": "mlx",
    "dimension": 384,
    "pooling": "mean",
    "normalized": True,
    "fingerprint": "sha256:" + "a" * 64,
}


def test_model_info_accepts_full_embedding_contract():
    info = ModelInfo(**VALID_MODEL)
    assert info.dimension == 384
    assert info.normalized is True


def test_model_info_accepts_summary_shape_without_dimension():
    info = ModelInfo(
        model="Qwen/Qwen2.5-Coder-0.5B-Instruct",
        revision="r2",
        backend="mlx",
        quantization="4bit",
        fingerprint="sha256:" + "b" * 64,
    )
    assert info.dimension is None


def test_model_info_requires_fingerprint():
    with pytest.raises(Exception):
        ModelInfo(model="m", revision="r", backend="mlx")


def test_model_info_rejects_bad_fingerprint_format():
    with pytest.raises(Exception):
        ModelInfo(**{**VALID_MODEL, "fingerprint": "not-a-hash"})


def test_model_info_rejects_unknown_backend():
    with pytest.raises(Exception):
        ModelInfo(**{**VALID_MODEL, "backend": "cuda"})


def test_embed_response_shape():
    response = EmbedResponse(
        vectors=[[0.1, 0.2]], dimension=2, fingerprint="sha256:" + "c" * 64
    )
    assert len(response.vectors) == 1
    assert response.dimension == 2


def test_summary_response_shape():
    response = SummaryResponse(
        summary="a summary", fingerprint="sha256:" + "d" * 64
    )
    assert response.summary == "a summary"
