"""relationships table contract tests: source-grounded CHECK constraints
(plan §5.2)."""

from __future__ import annotations

import sqlite3

import pytest

from tests.conftest import insert_file, insert_function, insert_module


def rel_row(**overrides):
    base = dict(
        source_type="function",
        source_id=1,
        relation="calls",
        target_type="function",
        target_id=2,
        target_name="helper",
        resolution="syntactic_candidate",
        snippet="do_work(helper())",
    )
    base.update(overrides)
    return base


@pytest.fixture()
def seeded(db):
    m = insert_module(db)
    f = insert_file(db, m)
    caller = insert_function(db, f, name="caller")
    callee = insert_function(db, f, name="helper")
    return m, f, caller, callee


def insert_rel(db, **overrides):
    params = rel_row(**overrides)
    params.update(overrides)
    params.setdefault("line_number", 1)
    params.setdefault("origin_file_id", 1)
    db.execute(
        "INSERT INTO relationships (source_type, source_id, relation, target_type,"
        " target_id, target_name, origin_file_id, line_number, resolution, snippet)"
        " VALUES (:source_type, :source_id, :relation, :target_type, :target_id,"
        " :target_name, :origin_file_id, :line_number, :resolution, :snippet)",
        params,
    )


def test_valid_resolved_relationship(db, seeded):
    _, f, caller, callee = seeded
    insert_rel(
        db, source_id=caller, target_id=callee, origin_file_id=f, line_number=3
    )
    assert db.execute("SELECT COUNT(*) FROM relationships").fetchone()[0] == 1


def test_unresolved_edge_with_name_only(db, seeded):
    _, f, caller, _ = seeded
    insert_rel(
        db,
        source_id=caller,
        target_id=None,
        target_type=None,
        target_name="external_thing",
        resolution="unresolved",
        origin_file_id=f,
    )
    assert db.execute("SELECT COUNT(*) FROM relationships").fetchone()[0] == 1


def test_rejects_target_without_name_or_id(db, seeded):
    _, f, caller, _ = seeded
    with pytest.raises(sqlite3.IntegrityError):
        insert_rel(
            db,
            source_id=caller,
            target_id=None,
            target_type=None,
            target_name=None,
            origin_file_id=f,
        )


def test_rejects_id_without_type(db, seeded):
    _, f, caller, callee = seeded
    with pytest.raises(sqlite3.IntegrityError):
        insert_rel(
            db,
            source_id=caller,
            target_id=callee,
            target_type=None,
            origin_file_id=f,
        )


def test_rejects_unknown_relation(db, seeded):
    _, f, caller, _ = seeded
    with pytest.raises(sqlite3.IntegrityError):
        insert_rel(db, source_id=caller, relation="teleports", origin_file_id=f)


def test_origin_file_cascade(db, seeded):
    _, f, caller, _ = seeded
    insert_rel(db, source_id=caller, origin_file_id=f)
    db.execute("DELETE FROM files WHERE id = ?", (f,))
    assert db.execute("SELECT COUNT(*) FROM relationships").fetchone()[0] == 0
