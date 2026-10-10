"""Worker HTTP client contract (plan §4.2–4.4).

The same versioned, bearer-authenticated HTTP contract is used for managed
loopback workers and remote workers. Verified against the fake worker over
a real socket.
"""

import os
import subprocess
import sys
import time

import pytest

from indexedcode.workers.client import (
    WorkerAuthError,
    WorkerClient,
    WorkerTransientError,
    classify_status,
    is_retryable,
)
from indexedcode.workers.fingerprint import model_fingerprint
from indexedcode.workers.protocol import ModelInfo

from tests.conftest import (
    FAKE_FINGERPRINT,
    FAKE_WORKER,
    free_port,
    wait_until,
)

SECRET = "test-secret-0123456789abcdef"
EXPECTED_FP = model_fingerprint(
    model="fake-embed", revision="r1", backend="mlx", dimension=3,
)


@pytest.fixture()
def fake_worker():
    """A running fake worker process bound to a free loopback port."""
    port = free_port()
    env = dict(
        os.environ,
        INDEXEDCODE_WORKER_PORT=str(port),
        INDEXEDCODE_WORKER_SECRET=SECRET,
        INDEXEDCODE_WORKER_TEST_FP=EXPECTED_FP,
    )
    proc = subprocess.Popen(
        [sys.executable, FAKE_WORKER, "--mode", "ready"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = WorkerClient(
        base_url=f"http://127.0.0.1:{port}",
        secret=SECRET,
        expected_fingerprint=EXPECTED_FP,
        timeout_s=10.0,
    )
    try:
        assert wait_until(
            lambda: _safe(client.model_info), timeout=10
        ), "fake worker never became ready"
        yield client
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _safe(call):
    try:
        return call()
    except Exception:
        return None


def test_model_info_parses_full_contract(fake_worker):
    info = fake_worker.model_info()
    assert isinstance(info, ModelInfo)
    assert info.model == "fake-embed"
    assert info.revision == "r1"
    assert info.backend == "mlx"
    assert info.dimension == 3
    assert info.fingerprint == EXPECTED_FP


def test_model_info_fingerprint_mismatch_raises(fake_worker):
    client = WorkerClient(
        base_url=fake_worker.base_url, secret=SECRET,
        expected_fingerprint="sha256:other", timeout_s=10.0,
    )
    from indexedcode.workers.client import FingerprintMismatchError

    with pytest.raises(FingerprintMismatchError):
        client.model_info()


def test_wrong_secret_is_an_auth_error(fake_worker):
    client = WorkerClient(
        base_url=fake_worker.base_url, secret="wrong-secret",
        expected_fingerprint=EXPECTED_FP, timeout_s=10.0,
    )
    with pytest.raises(WorkerAuthError):
        client.model_info()


def test_embed_returns_vectors_matching_request_size(fake_worker):
    response = fake_worker.embed(["alpha", "beta", "gamma"])
    assert len(response.vectors) == 3
    assert response.dimension == 3
    assert response.fingerprint == EXPECTED_FP


def test_summarize_returns_text_and_fingerprint(fake_worker):
    response = fake_worker.summarize(
        prompt="summarize this", max_output_tokens=64
    )
    assert response.summary == "fake summary"
    assert response.fingerprint == EXPECTED_FP


def test_server_error_is_classified_transient():
    assert classify_status(503) == "transient"
    assert classify_status(500) == "transient"
    assert classify_status(429) == "transient"
    assert classify_status(401) == "fatal"
    assert classify_status(404) == "fatal"
    assert classify_status(413) == "fatal"
    assert is_retryable(503)
    assert not is_retryable(401)


def test_overloaded_embed_raises_transient_error(tmp_path):
    port = free_port()
    env = dict(
        os.environ,
        INDEXEDCODE_WORKER_PORT=str(port),
        INDEXEDCODE_WORKER_SECRET=SECRET,
        INDEXEDCODE_WORKER_TEST_FP=EXPECTED_FP,
        INDEXEDCODE_WORKER_EMBED_STATUS="503",
    )
    proc = subprocess.Popen(
        [sys.executable, FAKE_WORKER, "--mode", "ready"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    client = WorkerClient(
        base_url=f"http://127.0.0.1:{port}", secret=SECRET,
        expected_fingerprint=EXPECTED_FP, timeout_s=10.0,
    )
    try:
        assert wait_until(lambda: _safe(client.model_info), timeout=10)
        with pytest.raises(WorkerTransientError):
            client.embed(["x"])
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_connection_refused_is_transient_not_fatal():
    port = free_port()  # nothing is listening
    client = WorkerClient(
        base_url=f"http://127.0.0.1:{port}", secret=SECRET,
        expected_fingerprint=EXPECTED_FP, timeout_s=1.0,
    )
    with pytest.raises(WorkerTransientError):
        client.model_info()
