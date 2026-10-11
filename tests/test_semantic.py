"""Semantic indexing wiring: CPU backend, worker identity, the embedder
callbacks, embedding backfill, fingerprint-scoped search, and the CLI."""

import json
import logging
import sys
import threading
import urllib.request

import pytest

from indexedcode import embeddings, retrieval, semantic
from indexedcode.cli import main
from indexedcode.config import DEFAULTS, ConfigError, load_config
from indexedcode.dal import enqueue_job
from indexedcode.http_api import create_server
from indexedcode.pipeline import process_next_job, run_until_idle
from indexedcode.reconcile import reconcile
from indexedcode.workers import hardwares, serve
from indexedcode.workers.adapters import backend_available, default_backend
from indexedcode.workers.errors import BackendUnavailableError
from indexedcode.workers.factory import _expected_fingerprint
from indexedcode.workers.fingerprint import model_fingerprint
from indexedcode.workers.protocol import EmbedResponse

from tests.conftest import FakeEmbedder, fake_vector, write_repo_sources

FP_A = "sha256:" + "a" * 64
FP_B = "sha256:" + "b" * 64

SRC = "def foo():\n    return 1\n\n\ndef bar():\n    return 2\n"


# --- backend selection -------------------------------------------------------


def test_cpu_backend_is_available_everywhere():
    assert backend_available("cpu", system="linux", machine="x86_64")
    assert backend_available("cpu", system="darwin", machine="arm64")


def test_unknown_backend_still_refused():
    with pytest.raises(BackendUnavailableError):
        backend_available("tpu")


@pytest.mark.parametrize(("system", "machine", "expected"), [
    ("darwin", "arm64", "mlx"),
    ("darwin", "aarch64", "mlx"),
    ("darwin", "x86_64", "cpu"),
    ("linux", "x86_64", "cpu"),
    ("linux", "arm64", "cpu"),
])
def test_default_backend_is_mlx_on_apple_silicon_else_cpu(
        system, machine, expected):
    assert default_backend(system, machine) == expected


def test_config_defaults_follow_platform_and_accept_cpu(tmp_path):
    assert DEFAULTS["embedding"]["backend"] == default_backend()
    path = tmp_path / "c.toml"
    path.write_text('[embedding]\nbackend = "cpu"\n')
    assert load_config(path)["embedding"]["backend"] == "cpu"
    path.write_text('[embedding]\nbackend = "tpu"\n')
    with pytest.raises(ConfigError):
        load_config(path)


# --- cpu adapters ------------------------------------------------------------


def test_normalize_returns_unit_vector():
    vec = hardwares._normalize([3.0, 4.0])
    assert vec == pytest.approx([0.6, 0.8])


def test_cpu_embed_without_runtime_raises_clear_error(monkeypatch):
    hardwares._cpu_embed_model.cache_clear()
    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(RuntimeError, match="CPU runtime"):
        hardwares.cpu_embed(["x"])


def test_cpu_summary_without_runtime_raises_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(RuntimeError, match="CPU runtime"):
        hardwares.cpu_summary("prompt")


def test_cpu_embed_mean_pools_and_normalizes(monkeypatch):
    """Drive cpu_embed with tiny stand-ins for torch/transformers."""
    torch = pytest.importorskip("torch")
    hardwares._cpu_embed_model.cache_clear()

    class Tok:
        def __call__(self, texts, **kw):
            n = len(texts)
            return {
                "attention_mask": torch.ones(n, 2, dtype=torch.long),
            }

    class Out:
        def __init__(self, n):
            self.last_hidden_state = torch.tensor(
                [[[3.0, 0.0], [3.0, 0.0]]] * n)

    class Model:
        def __call__(self, attention_mask):
            return Out(attention_mask.shape[0])

    monkeypatch.setattr(hardwares, "_cpu_embed_model",
                        lambda path: (Model(), Tok()))
    assert hardwares.cpu_embed(["a", "b"]) == pytest.approx(
        [[1.0, 0.0], [1.0, 0.0]])


# --- worker identity ---------------------------------------------------------


def test_build_model_info_embedding_and_summary():
    emb = serve.build_model_info("embedding", "cpu", "m")
    assert emb["backend"] == "cpu"
    assert emb["dimension"] == serve.DEFAULT_DIMENSION
    assert emb["normalized"] is True
    summ = serve.build_model_info("summary", "cpu", "m")
    assert "dimension" not in summ and "quantization" in summ


@pytest.mark.parametrize("backend", ["cpu", "mlx", "rocm"])
def test_managed_expected_fingerprint_matches_what_worker_advertises(backend):
    cfg = {"mode": "managed", "backend": backend, "model": "m"}
    advertised = model_fingerprint(
        **serve.build_model_info("embedding", backend, "m"))
    assert _expected_fingerprint(cfg, "embedding") == advertised


def test_explicit_expected_fingerprint_wins():
    cfg = {"mode": "managed", "backend": "cpu", "model": "m",
           "expected_fingerprint": FP_A}
    assert _expected_fingerprint(cfg, "embedding") == FP_A


def test_worker_main_passes_configured_model_to_runner(monkeypatch):
    seen = {}

    def runner(texts, model_path):
        seen["model_path"] = model_path
        return [[1.0]]

    captured = {}

    class StubServer:
        def __init__(self, addr, handler):
            captured["handler"] = handler

        def serve_forever(self):
            pass

    def fake_build_handler(secret, info, summary_fn, embed_fn):
        captured.update(info=info, embed_fn=embed_fn)
        return "handler"

    monkeypatch.setitem(serve._EMBED_RUNNERS, "cpu", runner)
    monkeypatch.setattr(serve, "LoopbackHTTPServer", StubServer)
    monkeypatch.setattr(serve, "build_handler", fake_build_handler)
    rc = serve.main(["--role", "embedding", "--backend", "cpu",
                     "--model", "my/model", "--port", "0"])
    assert rc == 0
    captured["embed_fn"](["x"])
    assert seen["model_path"] == "my/model"
    assert captured["info"]["fingerprint"] == model_fingerprint(
        **serve.build_model_info("embedding", "cpu", "my/model"))


def test_worker_main_selects_cpu_summary_runner(monkeypatch):
    captured = {}

    class StubServer:
        def __init__(self, *a):
            pass

        def serve_forever(self):
            pass

    monkeypatch.setattr(serve, "LoopbackHTTPServer", StubServer)
    monkeypatch.setattr(
        serve, "build_handler",
        lambda secret, info, summary_fn, embed_fn:
        captured.update(summary_fn=summary_fn, embed_fn=embed_fn))
    serve.main(["--role", "summary", "--backend", "cpu", "--model", "m",
                "--port", "0"])
    assert captured["summary_fn"].func is hardwares.cpu_summary
    assert captured["embed_fn"] is None


# --- SemanticIndex -----------------------------------------------------------


class StubClient:
    def __init__(self, vectors=None, dimension=3, fingerprint=FP_A):
        self.calls = []
        self.vectors = vectors
        self.dimension = dimension
        self.fingerprint = fingerprint
        self.timeout_s = 0

    def embed(self, texts):
        self.calls.append(list(texts))
        vectors = (self.vectors if self.vectors is not None
                   else [fake_vector(t) for t in texts])
        return EmbedResponse(vectors=vectors, dimension=self.dimension,
                             fingerprint=self.fingerprint)

    def model_info(self):
        class Info:
            fingerprint = FP_A
        return Info()


def test_embed_batches_by_max_batch_items():
    client = StubClient()
    index = semantic.SemanticIndex(None, client, FP_A, max_batch_items=2)
    vectors = index.embed(["a", "b", "c", "d", "e"])
    assert [len(c) for c in client.calls] == [2, 2, 1]
    assert len(vectors) == 5


def test_embed_empty_makes_no_request():
    client = StubClient()
    index = semantic.SemanticIndex(None, client, FP_A, 4)
    assert index.embed([]) == []
    assert client.calls == []


def test_embed_rejects_wrong_vector_count():
    index = semantic.SemanticIndex(
        None, StubClient(vectors=[[1.0, 0.0, 0.0]]), FP_A, 8)
    with pytest.raises(embeddings.EmbeddingValidationError):
        index.embed(["a", "b"])


def test_embed_rejects_unnormalized_vector():
    index = semantic.SemanticIndex(
        None, StubClient(vectors=[[2.0, 0.0, 0.0]]), FP_A, 8)
    with pytest.raises(embeddings.EmbeddingValidationError):
        index.embed(["a"])


def test_poll_and_stop_delegate_to_worker_when_present():
    class Worker:
        polled = stopped = 0

        def poll(self):
            self.polled += 1

        def stop(self):
            self.stopped += 1

    worker = Worker()
    index = semantic.SemanticIndex(worker, StubClient(), FP_A, 4)
    index.poll()
    index.stop()
    assert (worker.polled, worker.stopped) == (1, 1)
    bare = semantic.SemanticIndex(object(), StubClient(), FP_A, 4)
    bare.poll()
    bare.stop()  # no-ops for workers without supervision


class RemoteStub:
    """Looks like RemoteWorker: a client, no start()."""

    def __init__(self):
        self.client = StubClient()


class ManagedStub:
    """Looks like a started ManagedWorker."""

    def __init__(self, client_fails=False):
        self.started = self.stopped = False
        self.base_url = "http://127.0.0.1:1"
        self.secret = "s"
        self.expected_fingerprint = FP_A
        self.startup_timeout_s = 0

        class Info:
            fingerprint = FP_A
        self.model_info = Info()

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


def test_start_semantic_index_remote_worker(monkeypatch):
    monkeypatch.setattr(semantic, "build_worker",
                        lambda name, cfg: RemoteStub())
    index = semantic.start_semantic_index()
    assert index.fingerprint == FP_A
    assert index.client.timeout_s == semantic.EMBED_TIMEOUT_S
    assert index.client.calls == [["warmup"]]


def test_start_semantic_index_managed_worker_warms_up(monkeypatch):
    worker = ManagedStub()
    monkeypatch.setattr(semantic, "build_worker", lambda n, c: worker)
    monkeypatch.setattr(semantic, "WorkerClient",
                        lambda **kw: StubClient())
    index = semantic.start_semantic_index()
    assert worker.started and not worker.stopped
    assert worker.startup_timeout_s == semantic.STARTUP_TIMEOUT_S
    assert index.fingerprint == FP_A


def test_start_semantic_index_stops_worker_when_warmup_fails(monkeypatch):
    worker = ManagedStub()

    class Broken(StubClient):
        def embed(self, texts):
            raise RuntimeError("no runtime")

    monkeypatch.setattr(semantic, "build_worker", lambda n, c: worker)
    monkeypatch.setattr(semantic, "WorkerClient", lambda **kw: Broken())
    with pytest.raises(RuntimeError, match="no runtime"):
        semantic.start_semantic_index()
    assert worker.stopped


def test_start_semantic_index_reads_config_file(monkeypatch, tmp_path):
    seen = {}
    path = tmp_path / "c.toml"
    path.write_text('[embedding]\nbackend = "cpu"\nmodel = "my/m"\n'
                    "max_batch_items = 7\n")
    monkeypatch.setattr(
        semantic, "build_worker",
        lambda name, cfg: seen.update(cfg=cfg) or RemoteStub())
    index = semantic.start_semantic_index(path)
    assert seen["cfg"]["model"] == "my/m"
    assert index.max_batch_items == 7


def test_try_start_disabled_returns_none_without_starting(monkeypatch, caplog):
    monkeypatch.setattr(semantic, "start_semantic_index",
                        lambda p: pytest.fail("must not start"))
    with caplog.at_level(logging.INFO):
        assert semantic.try_start_semantic_index(enabled=False) is None
    assert "disabled" in caplog.text


def test_try_start_degrades_with_warning_on_failure(monkeypatch, caplog):
    def boom(path):
        raise RuntimeError("runtime missing")

    monkeypatch.setattr(semantic, "start_semantic_index", boom)
    with caplog.at_level(logging.WARNING):
        assert semantic.try_start_semantic_index() is None
    assert "runtime missing" in caplog.text
    assert "symbol/lexical" in caplog.text


def test_try_start_success_returns_index(monkeypatch):
    sentinel = semantic.SemanticIndex(None, StubClient(), FP_A, 4)
    monkeypatch.setattr(semantic, "start_semantic_index", lambda p: sentinel)
    assert semantic.try_start_semantic_index() is sentinel


# --- pipeline / reconcile backfill ------------------------------------------


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text(SRC)
    return tmp_path


def fingerprints(conn):
    return sorted(r["model_fingerprint"] for r in conn.execute(
        "SELECT model_fingerprint FROM entity_index"))


def index_with(conn, root, embed, fingerprint, now_ms=1000):
    reconcile(conn, str(root), now_ms=now_ms, model_fingerprint=fingerprint)
    return run_until_idle(conn, str(root), embed=embed, now_ms=now_ms,
                          model_fingerprint=fingerprint)


def test_run_until_idle_stamps_rows_with_model_fingerprint(db, repo):
    index_with(db, repo, FakeEmbedder(), FP_A)
    assert fingerprints(db) == [FP_A, FP_A]


def test_reconcile_backfills_ready_files_lacking_vectors(db, repo):
    # first index with no embedder: entities exist, no vectors
    reconcile(db, str(repo), now_ms=1000)
    run_until_idle(db, str(repo), now_ms=1000)
    assert fingerprints(db) == []

    again = reconcile(db, str(repo), now_ms=2000)
    assert again.refresh_enqueued == 0  # not requested without a model

    result = reconcile(db, str(repo), now_ms=3000, model_fingerprint=FP_A)
    assert result.refresh_enqueued == 1
    embed = FakeEmbedder()
    run_until_idle(db, str(repo), embed=embed, now_ms=3000,
                   model_fingerprint=FP_A)
    assert fingerprints(db) == [FP_A, FP_A]
    assert len(embed.calls) == 1 and len(embed.calls[0]) == 2

    done = reconcile(db, str(repo), now_ms=4000, model_fingerprint=FP_A)
    assert done.refresh_enqueued == 0 and done.unchanged == 1


def test_backfill_detected_after_cheap_filter_mismatch(db, repo):
    """mtime changed but content identical: still backfills if no vectors."""
    reconcile(db, str(repo), now_ms=1000)
    run_until_idle(db, str(repo), now_ms=1000)
    path = repo / "src" / "a.py"
    path.write_text(SRC)  # same bytes, new mtime
    import os
    st = path.stat()
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    result = reconcile(db, str(repo), now_ms=2000, model_fingerprint=FP_A)
    assert result.refresh_enqueued == 1


def test_model_change_reembeds_unchanged_entities(db, repo):
    index_with(db, repo, FakeEmbedder(), FP_A)
    embed = FakeEmbedder()
    result = reconcile(db, str(repo), now_ms=5000, model_fingerprint=FP_B)
    assert result.refresh_enqueued == 1
    run_until_idle(db, str(repo), embed=embed, now_ms=5000,
                   model_fingerprint=FP_B)
    assert fingerprints(db) == [FP_B, FP_B]
    assert len(embed.calls[0]) == 2


def test_unchanged_entities_with_current_model_are_not_reembedded(db, repo):
    index_with(db, repo, FakeEmbedder(), FP_A)
    embed = FakeEmbedder()
    enqueue_job(db, command="refresh_file", folder="src", file="a.py",
                now_ms=2000)
    outcome = process_next_job(db, str(repo), embed=embed, now_ms=2001,
                               model_fingerprint=FP_A)
    assert outcome == "completed"
    assert embed.calls == []


def test_retained_vector_keeps_its_original_model_identity(db, repo):
    index_with(db, repo, FakeEmbedder(), FP_A)
    (repo / "src" / "a.py").write_text(SRC.replace("return 2", "return 3"))
    enqueue_job(db, command="refresh_file", folder="src", file="a.py",
                now_ms=2000)
    # no embedder available now: old vectors survive, labelled as before
    outcome = process_next_job(db, str(repo), now_ms=2001,
                               model_fingerprint="unpinned")
    assert outcome == "completed"
    assert fingerprints(db) == [FP_A, FP_A]


# --- retrieval / http --------------------------------------------------------


def test_semantic_search_ignores_vectors_from_other_models(db, repo):
    index_with(db, repo, FakeEmbedder(), FP_A)
    embed = FakeEmbedder()
    other = retrieval.search(db, str(repo), "zzz", mode="semantic",
                             embedder=embed, model_fingerprint=FP_B)
    assert other["results"] == []
    same = retrieval.search(db, str(repo), "zzz", mode="semantic",
                            embedder=embed, model_fingerprint=FP_A)
    assert len(same["results"]) == 2
    unscoped = retrieval.search(db, str(repo), "zzz", mode="semantic",
                                embedder=embed)
    assert len(unscoped["results"]) == 2


def test_http_search_uses_embedder_and_reports_not_degraded(tmp_path):
    from tests.conftest import index_repo
    from indexedcode.db import connect, initialize

    write_repo_sources(tmp_path)
    conn = connect(tmp_path / "i.sqlite", writer=True, threaded=True)
    initialize(conn)
    index_repo(conn, tmp_path)
    embed = FakeEmbedder()
    httpd = create_server(conn, str(tmp_path), embedder=embed,
                          model_fingerprint="unpinned")
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        def post(payload):
            req = urllib.request.Request(
                httpd.base_url + "/v1/search",
                data=json.dumps(payload).encode(), method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return json.loads(resp.read())

        body = post({"query": "load configuration", "mode": "hybrid"})
        assert body["degraded"] is None
        assert embed.calls[-1] == ["load configuration"]
        assert body["results"]
        sem = post({"query": "anything", "mode": "semantic"})
        assert sem["results"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        conn.close()


def test_http_search_without_embedder_is_still_degraded(indexed_repo):
    conn, root = indexed_repo
    httpd = create_server(conn, str(root))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(
            httpd.base_url + "/v1/search",
            data=json.dumps({"query": "parse_config"}).encode(),
            method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert json.loads(resp.read())["degraded"] == "semantic_unavailable"
    finally:
        httpd.shutdown()
        httpd.server_close()


# --- CLI ---------------------------------------------------------------------


class FakeSemantic:
    fingerprint = FP_A

    def __init__(self):
        self.embedder = FakeEmbedder()
        self.stopped = False

    def embed(self, texts):
        return self.embedder(texts)

    def stop(self):
        self.stopped = True


@pytest.fixture()
def cli_semantic(monkeypatch):
    fake = FakeSemantic()
    calls = []

    def starter(config_path=None, enabled=True):
        calls.append({"config": config_path, "enabled": enabled})
        return fake if enabled else None

    monkeypatch.setattr("indexedcode.cli.try_start_semantic_index", starter)
    fake.calls = calls
    return fake


def test_index_once_builds_vectors_and_stops_worker(
        tmp_path, capsys, cli_semantic):
    root = tmp_path / "repo"
    root.mkdir()
    write_repo_sources(root)
    db = tmp_path / "index.sqlite"
    assert main(["index", "once", "--root", str(root), "--db", str(db)]) == 0
    assert json.loads(capsys.readouterr().out)["files_ready"] == 3
    assert cli_semantic.stopped
    assert cli_semantic.embedder.calls

    from indexedcode.db import connect
    conn = connect(db)
    fps = {r["model_fingerprint"] for r in conn.execute(
        "SELECT model_fingerprint FROM entity_index")}
    assert fps == {FP_A}
    conn.close()


def test_index_once_backfills_an_existing_lexical_index(
        tmp_path, capsys, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    write_repo_sources(root)
    db = tmp_path / "index.sqlite"
    # first run: embeddings off (conftest default stub returns None)
    assert main(["index", "once", "--root", str(root), "--db", str(db)]) == 0
    capsys.readouterr()

    fake = FakeSemantic()
    monkeypatch.setattr("indexedcode.cli.try_start_semantic_index",
                        lambda config_path=None, enabled=True: fake)
    assert main(["index", "once", "--root", str(root), "--db", str(db)]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["files_refreshed"] == 3
    assert fake.embedder.calls


def test_no_embeddings_flag_is_forwarded(tmp_path, capsys, cli_semantic):
    root = tmp_path / "repo"
    root.mkdir()
    write_repo_sources(root)
    main(["index", "once", "--root", str(root),
          "--db", str(tmp_path / "i.sqlite"), "--no-embeddings",
          "--config", "x.toml"])
    assert cli_semantic.calls == [{"config": "x.toml", "enabled": False}]
    assert cli_semantic.embedder.calls == []


def test_search_cli_uses_embedder_for_hybrid_not_symbol(
        tmp_path, capsys, cli_semantic):
    root = tmp_path / "repo"
    root.mkdir()
    write_repo_sources(root)
    db = tmp_path / "index.sqlite"
    main(["index", "once", "--root", str(root), "--db", str(db)])
    capsys.readouterr()
    cli_semantic.calls.clear()
    cli_semantic.stopped = False

    main(["search", "parse_config", "--db", str(db), "--root", str(root)])
    out = json.loads(capsys.readouterr().out)
    assert out["degraded"] is None
    assert cli_semantic.stopped

    cli_semantic.calls.clear()
    main(["search", "parse_config", "--db", str(db), "--root", str(root),
          "--mode", "symbol"])
    capsys.readouterr()
    assert cli_semantic.calls == []  # symbol mode needs no model


def test_reconcile_logged_reports_candidate_counts(db, repo, caplog):
    from indexedcode.cli import _reconcile_logged

    with caplog.at_level(logging.INFO):
        result = _reconcile_logged(db, str(repo), "initial scan", FP_A)
    assert result.candidates == 1
    assert "scanned 1 candidate files" in caplog.text


def test_drain_passes_embedder_and_fingerprint(db, repo):
    from indexedcode.cli import _drain

    reconcile(db, str(repo), now_ms=0, model_fingerprint=FP_A)
    fake = FakeSemantic()
    _drain(db, str(repo), fake)
    assert fingerprints(db) == [FP_A, FP_A]
