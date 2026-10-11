"""Shared test fixtures for the Phase 1 IndexedCode store suite.

All tests run against a real SQLite database file in a per-test tmp
directory. Nothing at the database boundary is mocked.
"""

from __future__ import annotations

import os
import socket
import sqlite3
import struct
import time
import uuid
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _no_real_embedding_worker(monkeypatch):
    """CLI tests must not spawn a real model worker; tests that want
    semantic indexing patch ``indexedcode.cli.try_start_semantic_index``."""
    monkeypatch.setattr("indexedcode.cli.try_start_semantic_index",
                        lambda config_path=None, enabled=True: None)


@pytest.fixture()
def db(tmp_path):
    """An initialized writer connection to a fresh database file."""
    from indexedcode.db import connect, initialize

    path = tmp_path / "index.sqlite"
    conn = connect(path, writer=True)
    initialize(conn)
    yield conn
    conn.close()


@pytest.fixture()
def reader_factory(tmp_path):
    """Open additional reader connections to the same database."""
    path = tmp_path / "index.sqlite"

    def _open():
        from indexedcode.db import connect

        return connect(path, writer=False)

    return _open


def pack_embedding(values, little_endian=True):
    """Serialize floats to a float32 blob as required by entity_index."""
    fmt = "<" + "f" * len(values) if little_endian else ">" + "f" * len(values)
    return struct.pack(fmt, *values)


def insert_module(conn, name="repo", relative_path="", parent_module_id=None):
    """Insert a minimal module row and return its id."""
    path = relative_path or name
    conn.execute(
        "INSERT INTO modules (parent_module_id, name, relative_path, updated_at_ms)"
        " VALUES (?, ?, ?, 1)",
        (parent_module_id, name, path),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def insert_file(
    conn,
    module_id,
    folder="",
    file="a.py",
    language="python",
    status="ready",
):
    conn.execute(
        "INSERT INTO files (module_id, folder, file, language, status, updated_at_ms)"
        " VALUES (?, ?, ?, ?, ?, 1)",
        (module_id, folder, file, language, status),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def insert_function(conn, file_id, name="do_work", qualified_name=None):
    conn.execute(
        "INSERT INTO functions (file_id, name, qualified_name, signature,"
        " start_line, end_line, start_byte, end_byte, source, source_hash,"
        " updated_at_ms) VALUES (?, ?, ?, 'def do_work(): ...', 1, 2, 0, 10,"
        " 'def do_work(): ...', 'deadbeef', 1)",
        (file_id, name, qualified_name or name),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def new_lease_token():
    return uuid.uuid4().hex


def sqlite_version(conn: sqlite3.Connection):
    return conn.execute("SELECT sqlite_version()").fetchone()[0]


# ---------------------------------------------------------------------------
# Phase 3: fake-worker helpers (test-only infrastructure)
# ---------------------------------------------------------------------------

FAKE_WORKER = str(Path(__file__).parent / "fake_worker.py")
FAKE_FINGERPRINT = "sha256:" + "0" * 64


def free_port():
    """Reserve and release an ephemeral loopback TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def pid_alive(pid):
    """True if the process still exists (ignores zombie reaping state)."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_until(predicate, timeout=10.0, interval=0.05):
    """Poll predicate until it is truthy or timeout elapses; return value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = predicate()
        if result:
            return result
        time.sleep(interval)
    return None


# ---------------------------------------------------------------------------
# Phase 4: deterministic inference fakes and pipeline helpers
# ---------------------------------------------------------------------------

def fake_vector(text, dimensions=3):
    """A deterministic unit-norm vector derived from the text."""
    import hashlib
    import math

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [
        (b / 255.0) - 0.5
        for b in digest[:dimensions]
    ]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


class FakeEmbedder:
    """Counts calls so tests can prove changed-entity-only inference."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def __call__(self, texts):
        if self.error is not None:
            raise self.error
        self.calls.append(list(texts))
        return [fake_vector(text) for text in texts]


class FakeSummarizer:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def __call__(self, texts):
        if self.error is not None:
            raise self.error
        self.calls.append(list(texts))
        return [f"summary of {t[:24]}" for t in texts]


# ---------------------------------------------------------------------------
# Phase 5: a small indexed repository shared by retrieval/adapter tests
# ---------------------------------------------------------------------------

REPO_SOURCES = {
    "src/config_loader.py": (
        "import json\n"
        "\n"
        "\n"
        "def parse_config(text):\n"
        "    return json.loads(text)\n"
        "\n"
        "\n"
        "class Config:\n"
        "    def load(self, path):\n"
        "        return parse_config(open(path).read())\n"
    ),
    "src/reporting.py": (
        "def format_report(items):\n"
        '    return "\\n".join(str(item) for item in items)\n'
    ),
    "tests/test_config_loader.py": (
        "from src.config_loader import parse_config\n"
        "\n"
        "\n"
        "def test_parse_config():\n"
        "    assert parse_config('{}') == {}\n"
    ),
}


def write_repo_sources(root):
    for rel, text in REPO_SOURCES.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def index_repo(conn, root, now_ms=1000):
    """Reconcile and process every job until idle; returns the embedder."""
    from indexedcode.pipeline import process_next_job
    from indexedcode.reconcile import reconcile

    reconcile(conn, str(root), now_ms=now_ms)
    embed = FakeEmbedder()
    now = now_ms
    for _ in range(200):
        outcome = process_next_job(
            conn, str(root), embed=embed, summarize=FakeSummarizer(),
            now_ms=now, retry_base_ms=1, retry_cap_ms=1, jitter=lambda: 0,
        )
        if outcome is None:
            next_due = conn.execute(
                "SELECT MIN(available_at_ms) AS next FROM jobs"
                " WHERE status IN ('pending', 'retry')"
            ).fetchone()["next"]
            if next_due is None:
                return embed
            now = max(now + 1, next_due + 1)
        else:
            now += 1
    raise AssertionError("indexing never went idle")


@pytest.fixture()
def indexed_repo(tmp_path):
    """A fully indexed three-file repository plus its (thread-capable)
    writer connection, shared with the in-test HTTP adapter."""
    from indexedcode.db import connect, initialize

    write_repo_sources(tmp_path)
    conn = connect(tmp_path / "index.sqlite", writer=True, threaded=True)
    initialize(conn)
    index_repo(conn, tmp_path)
    yield conn, tmp_path
    conn.close()
