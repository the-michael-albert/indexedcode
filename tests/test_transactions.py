"""Atomic publication and rollback contract tests (plan §7.3): readers see
either the previous complete file snapshot or the next one, never a partial
replacement."""

from __future__ import annotations

from indexedcode import dal
from tests.conftest import insert_file, insert_function, insert_module


def make_function(name, start_byte):
    return dict(
        name=name,
        qualified_name=name,
        signature=f"def {name}(): ...",
        start_line=1,
        end_line=2,
        start_byte=start_byte,
        end_byte=start_byte + 10,
        source=f"def {name}(): ...",
        source_hash=f"hash-{name}",
    )


def test_publish_replaces_previous_snapshot_atomically(db):
    m = insert_module(db)
    f = insert_file(db, m, status="pending")
    insert_function(db, f, name="old_fn")  # prior known-good snapshot

    dal.enqueue_job(db, command="refresh_file", folder="", file="a.py",
                    desired_revision=1)
    job = dal.claim_next_job(db, now_ms=1000)

    dal.publish_file_refresh(
        db,
        job=job,
        file_row=dict(folder="", file="a.py", module_id=m),
        classes=[],
        functions=[make_function("new_fn_a", 0), make_function("new_fn_b", 20)],
        relationships=[],
        now_ms=2000,
    )
    names = {
        r[0]
        for r in db.execute("SELECT name FROM functions WHERE file_id = ?", (f,))
    }
    assert names == {"new_fn_a", "new_fn_b"}
    file_row = db.execute(
        "SELECT status, indexed_hash FROM files WHERE id = ?", (f,)
    ).fetchone()
    assert file_row["status"] == "ready"


def test_publish_failure_leaves_prior_snapshot(db):
    m = insert_module(db)
    f = insert_file(db, m, status="ready")
    insert_function(db, f, name="good_fn")

    dal.enqueue_job(db, command="refresh_file", folder="", file="a.py",
                    desired_revision=1)
    job = dal.claim_next_job(db, now_ms=1000)

    bad_function = make_function("bad_fn", 0)
    bad_function["start_line"] = 5
    bad_function["end_line"] = 2  # violates CHECK: end_line >= start_line

    try:
        dal.publish_file_refresh(
            db,
            job=job,
            file_row=dict(folder="", file="a.py", module_id=m),
            classes=[],
            functions=[bad_function],
            relationships=[],
            now_ms=2000,
        )
        raised = False
    except Exception:
        raised = True
    assert raised

    names = {
        r[0]
        for r in db.execute("SELECT name FROM functions WHERE file_id = ?", (f,))
    }
    assert names == {"good_fn"}  # last-good snapshot fully retained


def test_removal_tombstones_file_and_descendants(db):
    m = insert_module(db)
    f = insert_file(db, m, status="ready")
    fn = insert_function(db, f, name="doomed")
    dal.insert_entity_index(
        db,
        entity_type="function",
        entity_id=fn,
        search_text="doomed",
        embedding=b"\x00" * 16,
        dimensions=4,
        input_hash="h",
        model_fingerprint="fp",
    )
    dal.remove_file(db, folder="", file="a.py", now_ms=3000)
    row = db.execute("SELECT status FROM files WHERE id = ?", (f,)).fetchone()
    assert row["status"] == "deleted"
    assert db.execute("SELECT COUNT(*) FROM functions").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM entity_index").fetchone()[0] == 0
