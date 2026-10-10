"""Shared test fixtures for the Phase 1 IndexedCode store suite.

All tests run against a real SQLite database file in a per-test tmp
directory. Nothing at the database boundary is mocked.
"""

from __future__ import annotations

import sqlite3
import struct
import uuid

import pytest


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
