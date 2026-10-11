"""Hosted embedding/summarization API mode (``mode = "api"``)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from indexedcode.config import ConfigError, load_config
from indexedcode.workers.api import ApiEmbeddingClient, ApiSummaryClient
from indexedcode.workers.errors import WorkerAuthError, WorkerTransientError
from indexedcode.workers.factory import build_worker


class FakeApi:
    """Records requests; replies per path from ``script`` (callable)."""

    def __init__(self, script):
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(
                    (self.path,
                     {k.lower(): v for k, v in self.headers.items()}, body))
                status, payload, headers = script(self.path, body,
                                                  len(outer.requests))
                data = json.dumps(payload).encode()
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever,
                         daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def serve():
    servers = []

    def make(script):
        api = FakeApi(script)
        servers.append(api)
        return api

    yield make
    for api in servers:
        api.close()


def test_embeddings_are_normalized_and_reordered(serve):
    def script(path, body, n):
        rows = [{"index": 1, "embedding": [0.0, 4.0]},
                {"index": 0, "embedding": [3.0, 0.0]}]
        return 200, {"data": rows}, {}

    api = serve(script)
    client = ApiEmbeddingClient("openai", api.url, "k", "m", dimensions=2)
    response = client.embed(["a", "b"])
    assert response.vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert response.dimension == 2
    path, headers, body = api.requests[0]
    assert path == "/embeddings"
    assert headers["authorization"] == "Bearer k"
    assert body == {"model": "m", "input": ["a", "b"], "dimensions": 2}
    assert response.fingerprint == client.model_info().fingerprint


def test_summary_openai_and_anthropic(serve):
    def script(path, body, n):
        if path == "/chat/completions":
            return 200, {"choices": [{"message": {"content": " hi "}}]}, {}
        return 200, {"content": [{"type": "text", "text": "yo"}]}, {}

    api = serve(script)
    openai = ApiSummaryClient("openai", api.url, "k", "gpt")
    assert openai.summarize("p", 50).summary == " hi "
    claude = ApiSummaryClient("anthropic", api.url, "k", "claude")
    assert claude.summarize("p", 50).summary == "yo"
    _, headers, body = api.requests[1]
    assert headers["x-api-key"] == "k"
    assert headers["anthropic-version"]
    assert body["max_tokens"] == 50 and body["temperature"] == 0
    assert openai.model_info().fingerprint != claude.model_info().fingerprint


def test_429_is_retried_with_retry_after(serve):
    sleeps = []

    def script(path, body, n):
        if n < 3:
            return 429, {}, {"Retry-After": "2"}
        return 200, {"choices": [{"message": {"content": "ok"}}]}, {}

    api = serve(script)
    client = ApiSummaryClient("openai", api.url, "k", "m",
                              sleep=sleeps.append)
    assert client.summarize("p", 10).summary == "ok"
    assert sleeps == [2.0, 2.0]


def test_persistent_5xx_raises_transient(serve):
    api = serve(lambda *a: (503, {}, {}))
    client = ApiSummaryClient("openai", api.url, "k", "m", max_retries=1,
                              sleep=lambda s: None)
    with pytest.raises(WorkerTransientError):
        client.summarize("p", 10)
    assert len(api.requests) == 2


def test_bad_key_is_fatal_not_retried(serve):
    api = serve(lambda *a: (401, {}, {}))
    client = ApiSummaryClient("openai", api.url, "bad", "m")
    with pytest.raises(WorkerAuthError):
        client.summarize("p", 10)
    assert len(api.requests) == 1


def test_anthropic_cannot_embed():
    with pytest.raises(ValueError):
        ApiEmbeddingClient("anthropic", None, "k", "m")


def _write(tmp_path, text):
    path = tmp_path / "indexedcode.toml"
    path.write_text(text)
    return path


def test_config_api_defaults(tmp_path):
    cfg = load_config(_write(tmp_path, """
[embedding]
mode = "api"
provider = "openai"
[summarization]
mode = "api"
provider = "anthropic"
"""))
    assert cfg["embedding"]["model"] == "text-embedding-3-small"
    assert cfg["summarization"]["model"] == "claude-haiku-5-5"
    # no managed budget profile for hosted APIs; parallel by default
    assert cfg["summarization"]["max_concurrent_requests"] > 1


def test_config_api_requires_provider_and_secure_url(tmp_path):
    with pytest.raises(ConfigError, match="provider"):
        load_config(_write(tmp_path, '[embedding]\nmode = "api"\n'))
    with pytest.raises(ConfigError, match="https"):
        load_config(_write(tmp_path, """
[embedding]
mode = "api"
provider = "openai"
base_url = "http://api.example.com/v1"
"""))
    with pytest.raises(ConfigError, match="embedding models"):
        load_config(_write(tmp_path, """
[embedding]
mode = "api"
provider = "anthropic"
"""))


def test_factory_reads_key_from_env(monkeypatch):
    from indexedcode.workers.remote import RemoteWorker

    cfg = {"mode": "api", "provider": "openai", "model": "m",
           "base_url": None, "api_key_env": "MY_KEY", "dimensions": None}
    monkeypatch.delenv("MY_KEY", raising=False)
    with pytest.raises(ConfigError, match="MY_KEY"):
        build_worker("embedding", cfg)
    monkeypatch.setenv("MY_KEY", "secret")
    worker = build_worker("embedding", cfg)
    assert isinstance(worker, RemoteWorker)
    assert "secret" not in json.dumps(worker.status())


def test_semantic_index_runs_requests_concurrently_in_order():
    from types import SimpleNamespace

    from indexedcode.semantic import SemanticIndex

    barrier = threading.Barrier(3, timeout=5)

    class Client:
        def summarize(self, prompt, max_tokens):
            barrier.wait()  # only passes if 3 calls overlap
            return SimpleNamespace(summary=prompt.split("```\n")[1].strip())

    index = SemanticIndex(None, None, "fp", 32, summary_client=Client(),
                          summary_concurrency=3)
    texts = ["a", "b", "c"]
    out = index.summarize(texts)
    assert out == texts
