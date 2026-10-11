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
        ModelInfo(**{**VALID_MODEL, "backend": "tpu"})


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


def _serve_handler(secret="s"):
    import threading
    from http.server import HTTPServer

    from indexedcode.workers.serve import build_handler

    info = {"fingerprint": "sha256:" + "0" * 64}
    server = HTTPServer(("127.0.0.1", 0), build_handler(
        secret, info, lambda p, n: "sum", lambda texts: [[1.0]] * len(texts)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_healthz_is_unauthenticated_but_everything_else_is_not():
    import urllib.error
    import urllib.request

    server = _serve_handler()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        assert urllib.request.urlopen(base + "/healthz").status == 200
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(base + "/v1/model")
        assert err.value.code == 401
    finally:
        server.shutdown()
        server.server_close()


def test_worker_refuses_public_bind_without_secret(monkeypatch):
    from indexedcode.workers import serve

    monkeypatch.delenv("INDEXEDCODE_WORKER_SECRET", raising=False)
    monkeypatch.setattr(serve, "backend_available", lambda backend: True)
    with pytest.raises(SystemExit):
        serve.main(["--role", "embedding", "--backend", "cpu",
                    "--model", "m", "--host", "0.0.0.0"])
