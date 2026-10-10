"""entity_index contract tests: polymorphic association validated by the
writer inside the transaction, blob/dimension integrity, and fingerprint
grouping (plan §5.1, §5.2)."""

from __future__ import annotations

import sqlite3

import pytest

from indexedcode import dal
from indexedcode.dal import PolymorphicIntegrityError
from tests.conftest import (
    insert_file,
    insert_function,
    insert_module,
    pack_embedding,
)

FP = "minilm-v1@abc123"


def valid_kwargs(dim=4):
    return dict(
        search_text="function do_work",
        embedding=pack_embedding([0.5, 0.5, 0.5, 0.5]),
        dimensions=dim,
        input_hash="inhash1",
        model_fingerprint=FP,
    )


def test_insert_for_existing_function(db):
    m = insert_module(db)
    f = insert_file(db, m)
    fn = insert_function(db, f)
    dal.insert_entity_index(db, entity_type="function", entity_id=fn, **valid_kwargs())
    row = db.execute("SELECT * FROM entity_index").fetchone()
    assert row["entity_type"] == "function"
    assert row["entity_id"] == fn
    assert row["embedding_dimensions"] == 4
    assert row["model_fingerprint"] == FP


def test_rejects_orphan_entity_id(db):
    # No matching row in any entity table: the writer must refuse.
    with pytest.raises(PolymorphicIntegrityError):
        dal.insert_entity_index(
            db, entity_type="function", entity_id=4242, **valid_kwargs()
        )
    # Nothing was written.
    assert db.execute("SELECT COUNT(*) FROM entity_index").fetchone()[0] == 0


def test_rejects_cross_type_mismatch(db):
    m = insert_module(db)
    f = insert_file(db, m)
    fn = insert_function(db, f)
    # fn is a function id, not a class id.
    with pytest.raises(PolymorphicIntegrityError):
        dal.insert_entity_index(
            db, entity_type="class", entity_id=fn, **valid_kwargs()
        )


def test_rejects_wrong_type_string(db):
    m = insert_module(db)
    with pytest.raises(ValueError):
        dal.insert_entity_index(
            db, entity_type="variable", entity_id=m, **valid_kwargs()
        )


def test_unique_per_entity(db):
    m = insert_module(db)
    f = insert_file(db, m)
    fn = insert_function(db, f)
    dal.insert_entity_index(db, entity_type="function", entity_id=fn, **valid_kwargs())
    with pytest.raises(sqlite3.IntegrityError):
        dal.insert_entity_index(
            db, entity_type="function", entity_id=fn, **valid_kwargs()
        )


def test_reupsert_replaces_row(db):
    m = insert_module(db)
    f = insert_file(db, m)
    fn = insert_function(db, f)
    dal.insert_entity_index(db, entity_type="function", entity_id=fn, **valid_kwargs())
    changed = valid_kwargs()
    changed["search_text"] = "function do_work v2"
    changed["input_hash"] = "inhash2"
    dal.upsert_entity_index(
        db, entity_type="function", entity_id=fn, **changed
    )
    rows = db.execute(
        "SELECT search_text FROM entity_index WHERE entity_type='function'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "function do_work v2"


def test_embedding_length_must_match_dimensions(db):
    m = insert_module(db)
    with pytest.raises(sqlite3.IntegrityError):
        dal.insert_entity_index(
            db,
            entity_type="module",
            entity_id=m,
            search_text="m",
            embedding=pack_embedding([1.0, 2.0]),  # 2 floats, dims says 4
            dimensions=4,
            input_hash="h",
            model_fingerprint=FP,
        )


def test_delete_entities_for_file(db):
    m = insert_module(db)
    f = insert_file(db, m)
    fn1 = insert_function(db, f, name="f1")
    fn2 = insert_function(db, f, name="f2")
    for fn in (fn1, fn2):
        dal.insert_entity_index(
            db, entity_type="function", entity_id=fn, **valid_kwargs()
        )
    deleted = dal.delete_entity_index_for_file(db, file_id=f)
    assert deleted == 2
    assert db.execute("SELECT COUNT(*) FROM entity_index").fetchone()[0] == 0
