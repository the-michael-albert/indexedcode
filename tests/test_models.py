"""Contract tests for DAL object shapes and enums (pydantic).

The enums must mirror the SQL CHECK constraints in db.py exactly, and the
DAL must accept/return these typed models while staying compatible with the
existing call sites (Row-style subscript access, ValueError compatibility).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from indexedcode import dal
from indexedcode.models import (
    JOB_COMMANDS,
    JOB_STATUSES,
    ClassDef,
    ClassKind,
    EntityIndexInput,
    EntityType,
    FileStatus,
    FunctionDef,
    Job,
    JobStatus,
    MetricComponent,
    MetricKind,
    MetricOutcome,
    NewJob,
    Relation,
    RelationshipRecord,
    Resolution,
)
from tests.conftest import insert_file, insert_module, pack_embedding


# --- enums mirror the SQL CHECK constraints -------------------------------


@pytest.mark.parametrize(
    ("enum", "expected"),
    [
        ("JobCommand", {"refresh_file", "remove_file", "refresh_module"}),
        ("JobStatus", {"pending", "running", "retry", "failed", "done"}),
        ("FileStatus", {"pending", "ready", "error", "skipped", "deleted"}),
        ("ClassKind", {"class", "interface", "struct", "enum"}),
        ("EntityType", {"module", "file", "class", "function"}),
        ("Relation", {"imports", "calls", "inherits", "implements", "references"}),
        ("Resolution", {"unresolved", "syntactic_candidate", "resolved"}),
        ("MetricKind", {"request", "job"}),
        (
            "MetricOutcome",
            {"success", "client_error", "dependency_error", "retry", "failed",
             "cancelled"},
        ),
        (
            "MetricComponent",
            {"orchestrator", "embedding_worker", "summary_worker",
             "remote_embedding", "remote_summary"},
        ),
        ("Backend", {"mlx", "rocm", "remote"}),
    ],
)
def test_enum_members_match_sql_checks(enum, expected):
    import indexedcode.models as m

    values = {e.value for e in getattr(m, enum)}
    assert values == expected


def test_enum_constant_tuples_match_members():
    assert set(JOB_COMMANDS) == {"refresh_file", "remove_file", "refresh_module"}
    assert set(JOB_STATUSES) == {"pending", "running", "retry", "failed", "done"}


def test_unknown_enum_value_rejected():
    with pytest.raises(ValueError):
        JobStatus("paused")
    with pytest.raises(ValueError):
        EntityType("variable")


# --- input shapes ------------------------------------------------------------


def test_new_job_accepts_valid_input():
    job = NewJob(command="refresh_file", folder="pkg", file="a.py",
                 desired_revision=1)
    assert job.command == "refresh_file"


def test_new_job_rejects_bad_values():
    with pytest.raises(ValidationError):
        NewJob(command="explode", folder="pkg", desired_revision=1)
    with pytest.raises(ValidationError):
        NewJob(command="refresh_file", folder="pkg", desired_revision=0)
    with pytest.raises(ValidationError):
        NewJob(command="refresh_file", folder="pkg", line_number=0)
    with pytest.raises(ValidationError):
        NewJob(command="refresh_file")  # folder is required


def test_function_def_rejects_bad_coordinates():
    ok = dict(name="f", qualified_name="f", signature="f()",
              start_line=1, end_line=2, start_byte=0, end_byte=10,
              source="f()", source_hash="h")
    FunctionDef(**ok)
    with pytest.raises(ValidationError):
        FunctionDef(**{**ok, "end_line": 0})  # before start
    with pytest.raises(ValidationError):
        FunctionDef(**{**ok, "start_byte": -1})


def test_class_def_requires_known_kind():
    ok = dict(name="A", qualified_name="pkg.A", kind="class",
              start_line=1, end_line=5, start_byte=0, end_byte=50,
              source="class A: ...", source_hash="h")
    ClassDef(**ok)
    with pytest.raises(ValidationError):
        ClassDef(**{**ok, "kind": "trait"})


def test_relationship_record_validates_target_pairing():
    ok = dict(source_type="function", source_id=1, relation="calls",
              line_number=1, resolution="resolved", snippet="f(g())")
    RelationshipRecord(**ok, target_type="function", target_id=2,
                       target_name="g")
    # id without type is rejected by the model, not just by SQL
    with pytest.raises(ValidationError):
        RelationshipRecord(**ok, target_id=2)
    with pytest.raises(ValidationError):
        RelationshipRecord(**ok, target_type=None, target_id=None,
                           target_name=None)  # no target at all


def test_entity_index_input_requires_dimensions():
    with pytest.raises(ValidationError):
        EntityIndexInput(entity_type="function", entity_id=1,
                         search_text="x", embedding=b"\x00" * 16,
                         input_hash="h", model_fingerprint="fp")  # no dims


# --- DAL integration: typed inputs and return values -------------------------


def test_enqueue_returns_job_model(db):
    job = dal.enqueue_job(db, command="refresh_file", folder="pkg",
                          file="a.py", desired_revision=1)
    assert isinstance(job, Job)
    assert job.status == "pending"


def test_enqueue_rejects_invalid_command_via_model(db):
    with pytest.raises(ValueError):  # ValidationError is a ValueError
        dal.enqueue_job(db, command="explode", folder="pkg",
                        desired_revision=1)


def test_claim_returns_job_model_with_row_compatibility(db):
    dal.enqueue_job(db, command="refresh_file", folder="pkg", file="a.py",
                    desired_revision=1)
    claimed = dal.claim_next_job(db, now_ms=1000)
    assert isinstance(claimed, Job)
    # Row-style subscript access keeps existing call sites working.
    assert claimed["lease_token"]
    assert claimed["status"] == "running"
    assert claimed.attempt_count == 1


def test_enqueue_accepts_new_job_model(db):
    new = NewJob(command="remove_file", folder="pkg", file="gone.py",
                 desired_revision=3)
    job = dal.enqueue_job(db, new)
    assert isinstance(job, Job)
    assert job.command == "remove_file"
    assert job.desired_revision == 3


def test_publish_accepts_typed_definitions(db):
    m = insert_module(db)
    insert_file(db, m, status="pending")
    dal.enqueue_job(db, command="refresh_file", folder="", file="a.py",
                    desired_revision=1)
    job = dal.claim_next_job(db, now_ms=1000)
    dal.publish_file_refresh(
        db,
        job=job,
        file_row=dict(folder="", file="a.py", module_id=m),
        classes=[ClassDef(name="A", qualified_name="A", kind="class",
                          start_line=1, end_line=5, start_byte=0,
                          end_byte=50, source="class A: ...",
                          source_hash="h")],
        functions=[FunctionDef(name="f", qualified_name="f",
                               signature="f()", start_line=6, end_line=7,
                               start_byte=50, end_byte=60, source="f()",
                               source_hash="h")],
        relationships=[RelationshipRecord(
            source_type="function", source_id=1, relation="references",
            target_name="A", line_number=6,
            resolution="syntactic_candidate", snippet="A",
        )],
        now_ms=2000,
    )
    assert db.execute("SELECT COUNT(*) FROM classes").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM relationships").fetchone()[0] == 1


def test_publish_rejects_invalid_entity_before_sql(db):
    m = insert_module(db)
    insert_file(db, m, status="pending")
    dal.enqueue_job(db, command="refresh_file", folder="", file="a.py",
                    desired_revision=1)
    job = dal.claim_next_job(db, now_ms=1000)
    bad = dict(name="f", qualified_name="f", signature="f()",
               start_line=5, end_line=2, start_byte=0, end_byte=10,
               source="f()", source_hash="h")
    with pytest.raises(ValidationError):
        dal.publish_file_refresh(
            db, job=job,
            file_row=dict(folder="", file="a.py", module_id=m),
            classes=[], functions=[bad], relationships=[], now_ms=2000,
        )


def test_insert_entity_index_accepts_model(db):
    m = insert_module(db)
    f = insert_file(db, m)
    inp = EntityIndexInput(
        entity_type="module", entity_id=m, search_text="repo",
        embedding=pack_embedding([1.0, 0.0]), dimensions=2,
        input_hash="h", model_fingerprint="fp",
    )
    dal.insert_entity_index(db, inp)
    assert db.execute("SELECT COUNT(*) FROM entity_index").fetchone()[0] == 1
