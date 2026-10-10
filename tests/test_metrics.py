"""metrics contract tests: request/job row constraints, detail_json key
whitelist, best-effort semantics (a metrics failure must never alter a
committed result), and bounded retention cleanup (plan §5.4)."""

from __future__ import annotations

import sqlite3

import pytest

from indexedcode import dal
from indexedcode.dal import MetricValidationError
from tests.conftest import insert_file, insert_function, insert_module


def test_request_metric_roundtrip(db):
    dal.record_metric(
        db,
        metric_kind="request",
        operation="search",
        outcome="success",
        component="orchestrator",
        now_ms=1000,
        request_duration_ms=42,
        status_code=200,
        request_bytes=120,
        response_bytes=900,
    )
    row = db.execute("SELECT * FROM metrics").fetchone()
    assert row["metric_kind"] == "request"
    assert row["operation"] == "search"
    assert row["request_duration_ms"] == 42


def test_request_metric_requires_duration(db):
    with pytest.raises(sqlite3.IntegrityError):
        dal.record_metric(
            db,
            metric_kind="request",
            operation="search",
            outcome="success",
            component="orchestrator",
            now_ms=1000,
        )


def test_job_metric_roundtrip(db):
    m = insert_module(db)
    insert_file(db, m)
    job_id = dal.enqueue_job(
        db, command="refresh_file", folder="", file="a.py", desired_revision=1
    )
    dal.record_metric(
        db,
        metric_kind="job",
        operation="refresh_file",
        outcome="success",
        component="orchestrator",
        now_ms=5000,
        job_id=job_id,
        job_attempt=1,
        queue_duration_ms=100,
        process_duration_ms=250,
        detail={"entities_changed": 3, "stale_discarded": False},
    )
    row = db.execute("SELECT * FROM metrics").fetchone()
    assert row["job_id"] == job_id
    assert row["detail_json"] == '{"entities_changed": 3, "stale_discarded": false}'


def test_job_metric_requires_job_fields(db):
    job_id = dal.enqueue_job(
        db, command="refresh_file", folder="", file="a.py", desired_revision=1
    )
    with pytest.raises(sqlite3.IntegrityError):
        dal.record_metric(
            db,
            metric_kind="job",
            operation="refresh_file",
            outcome="success",
            component="orchestrator",
            now_ms=5000,
            job_id=job_id,
            job_attempt=1,
            # queue_duration_ms missing
            process_duration_ms=250,
        )


def test_detail_json_rejects_unapproved_keys(db):
    with pytest.raises(MetricValidationError):
        dal.record_metric(
            db,
            metric_kind="request",
            operation="search",
            outcome="success",
            component="orchestrator",
            now_ms=1000,
            request_duration_ms=5,
            detail={"file_path": "pkg/a.py"},  # paths are forbidden
        )


def test_detail_json_rejects_unapproved_string_values(db):
    with pytest.raises(MetricValidationError):
        dal.record_metric(
            db,
            metric_kind="request",
            operation="search",
            outcome="success",
            component="orchestrator",
            now_ms=1000,
            request_duration_ms=5,
            detail={"error_message": "boom at pkg/a.py"},
        )


def test_metrics_failure_never_rolls_back_committed_job(db, monkeypatch):
    m = insert_module(db)
    f = insert_file(db, m)
    fn = insert_function(db, f)
    job_id = dal.enqueue_job(
        db, command="refresh_file", folder="", file="a.py", desired_revision=1
    )
    claimed = dal.claim_next_job(db, now_ms=1000)

    def broken_metric(*args, **kwargs):
        raise MetricValidationError("detail rejected")

    monkeypatch.setattr(dal, "record_metric", broken_metric)

    # The publication path must tolerate the best-effort metrics failure.
    dal.publish_file_refresh(
        db,
        job=claimed,
        file_row=dict(folder="", file="a.py", module_id=m),
        classes=[],
        functions=[
            dict(
                name="do_work",
                qualified_name="do_work",
                signature="def do_work(): ...",
                start_line=1,
                end_line=2,
                start_byte=0,
                end_byte=10,
                source="def do_work(): ...",
                source_hash="deadbeef",
            )
        ],
        relationships=[],
        now_ms=2000,
    )
    # The refresh committed despite the broken metrics append.
    status = db.execute("SELECT status FROM files WHERE id = ?", (f,)).fetchone()[0]
    assert status == "ready"
    assert db.execute("SELECT COUNT(*) FROM functions").fetchone()[0] >= 1


def test_retention_cleanup_deletes_old_rows_in_batches(db):
    for i in range(10):
        dal.record_metric(
            db,
            metric_kind="request",
            operation="search",
            outcome="success",
            component="orchestrator",
            now_ms=i * 1000,
            request_duration_ms=1,
        )
    deleted = dal.prune_metrics(db, now_ms=10_000, max_age_ms=5_000, batch=4)
    assert deleted == 4  # rows recorded at 0,1000,2000,3000... aged > 5000ms
    remaining = db.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    assert remaining == 6
