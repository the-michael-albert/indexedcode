"""Schema-level contract tests: eight application tables, pragmas, and DDL
CHECK constraints, exactly as specified in IMPLEMENTATION_PLAN.md §5.2."""

from __future__ import annotations

import sqlite3

import pytest

from tests.conftest import (
    insert_file,
    insert_function,
    insert_module,
)

APP_TABLES = {
    "jobs",
    "modules",
    "files",
    "classes",
    "functions",
    "entity_index",
    "relationships",
    "metrics",
}


def test_creates_exactly_the_eight_application_tables(db):
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
        " AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    assert {r[0] for r in rows} == APP_TABLES


def test_foreign_keys_pragma_enabled_on_every_connection(db, reader_factory):
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    reader = reader_factory()
    try:
        assert reader.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        reader.close()


def test_wal_journal_mode(db):
    assert db.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"


def test_busy_timeout_configured(db):
    assert db.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_initialize_is_idempotent(db):
    from indexedcode.db import initialize

    initialize(db)  # must not raise or duplicate anything
    rows = db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    assert {r[0] for r in rows} == APP_TABLES


def test_files_folder_file_unique(db):
    m = insert_module(db)
    insert_file(db, m, folder="pkg", file="a.py")
    with pytest.raises(sqlite3.IntegrityError):
        insert_file(db, m, folder="pkg", file="a.py")


def test_file_cannot_reference_missing_module(db):
    with pytest.raises(sqlite3.IntegrityError):
        insert_file(db, module_id=99999)


def test_module_deletion_cascades_children_not_files(db):
    parent = insert_module(db, name="pkg", relative_path="pkg")
    child = insert_module(
        db, name="sub", relative_path="pkg/sub", parent_module_id=parent
    )
    file_id = insert_file(db, child, folder="pkg/sub", file="a.py")
    insert_function(db, file_id)

    # files has ON DELETE RESTRICT against modules, so deletion is refused
    with pytest.raises(sqlite3.IntegrityError):
        db.execute("DELETE FROM modules WHERE id = ?", (child,))


def test_file_deletion_cascades_functions(db):
    m = insert_module(db)
    f = insert_file(db, m)
    insert_function(db, f)
    db.execute("DELETE FROM files WHERE id = ?", (f,))
    assert db.execute("SELECT COUNT(*) FROM functions").fetchone()[0] == 0


def test_class_nesting_and_kind_constraint(db):
    m = insert_module(db)
    f = insert_file(db, m)
    db.execute(
        "INSERT INTO classes (file_id, name, qualified_name, kind, start_line,"
        " end_line, start_byte, end_byte, source, source_hash, updated_at_ms)"
        " VALUES (?, 'A', 'pkg.A', 'class', 1, 5, 0, 50, 'class A: ...',"
        " 'aa', 1)",
        (f,),
    )
    class_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO classes (file_id, parent_class_id, name, qualified_name,"
            " kind, start_line, end_line, start_byte, end_byte, source,"
            " source_hash, updated_at_ms)"
            " VALUES (?, ?, 'B', 'pkg.A.B', 'trait', 2, 4, 10, 40, '...', 'bb', 1)",
            (f, class_id),
        )


def test_line_coordinates_one_based_inclusive(db):
    m = insert_module(db)
    f = insert_file(db, m)
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO functions (file_id, name, qualified_name, signature,"
            " start_line, end_line, start_byte, end_byte, source, source_hash,"
            " updated_at_ms) VALUES (?, 'x', 'x', 'x()', 0, 2, 0, 5, 'x', 'h', 1)",
            (f,),
        )
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO functions (file_id, name, qualified_name, signature,"
            " start_line, end_line, start_byte, end_byte, source, source_hash,"
            " updated_at_ms) VALUES (?, 'x', 'x', 'x()', 5, 2, 0, 5, 'x', 'h', 1)",
            (f,),
        )
