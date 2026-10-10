"""Job queue contract tests: enqueue coalescing, lease/claim, retry with
backoff, reclaim of expired leases, and revision recheck (plan §5.1, §7.2)."""

from __future__ import annotations

import sqlite3

import pytest

from indexedcode import dal
from tests.conftest import new_lease_token


def get_job(db, job_id):
    row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    assert row is not None
    return row


def test_enqueue_creates_pending_job(db):
    job_id = dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=1
    )
    job = get_job(db, job_id)
    assert job["command"] == "refresh_file"
    assert job["folder"] == "pkg"
    assert job["file"] == "a.py"
    assert job["status"] == "pending"
    assert job["desired_revision"] == 1
    assert job["attempt_count"] == 0


def test_enqueue_rejects_unknown_command(db):
    with pytest.raises(ValueError):
        dal.enqueue_job(db, command="explode", folder="pkg", desired_revision=1)


def test_enqueue_rejects_nonpositive_revision(db):
    with pytest.raises(ValueError):
        dal.enqueue_job(
            db, command="refresh_file", folder="pkg", file="a.py", desired_revision=0
        )


def test_refresh_coalescing_keeps_one_job_and_latest_revision(db):
    dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=1
    )
    job_id = dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=7
    )
    count = db.execute(
        "SELECT COUNT(*) FROM jobs WHERE folder='pkg' AND file='a.py'"
    ).fetchone()[0]
    assert count == 1
    assert get_job(db, job_id)["desired_revision"] == 7


def test_coalescing_does_not_span_different_files(db):
    dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=1
    )
    dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="b.py", desired_revision=1
    )
    assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 2


def test_coalescing_does_not_touch_running_job(db):
    job_id = dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=1
    )
    claimed = dal.claim_next_job(db, now_ms=1000)
    assert claimed is not None and claimed["id"] == job_id
    # A watcher event arriving while the job runs must create a separate
    # pending job rather than mutating the running row.
    new_id = dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=2
    )
    assert new_id != job_id
    assert get_job(db, job_id)["status"] == "running"
    assert get_job(db, new_id)["status"] == "pending"


def test_claim_returns_earliest_due_job_and_sets_lease(db):
    dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=1
    )
    claimed = dal.claim_next_job(db, now_ms=5000)
    assert claimed is not None
    assert claimed["status"] == "running"
    assert claimed["lease_token"]
    assert claimed["lease_expires_at_ms"] > 5000
    assert claimed["attempt_count"] == 1


def test_claim_skips_not_yet_available_jobs(db):
    dal.enqueue_job(
        db,
        command="refresh_file",
        folder="pkg",
        file="a.py",
        desired_revision=1,
        available_at_ms=9000,
    )
    assert dal.claim_next_job(db, now_ms=5000) is None


def test_claim_ignores_retry_row_before_backoff_elapses(db):
    job_id = dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=1
    )
    claimed = dal.claim_next_job(db, now_ms=1000)
    dal.retry_job(
        db,
        job_id=job_id,
        lease_token=claimed["lease_token"],
        last_error="dependency unavailable",
        retry_at_ms=10_000,
        now_ms=1100,
    )
    assert dal.claim_next_job(db, now_ms=9999) is None
    again = dal.claim_next_job(db, now_ms=10_000)
    assert again is not None and again["id"] == job_id
    assert again["status"] == "running"
    assert again["attempt_count"] == 2


def test_running_job_requires_lease(db):
    # The DDL CHECK must reject a running row without a lease token.
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO jobs (command, folder, status, desired_revision,"
            " available_at_ms, created_at_ms, updated_at_ms)"
            " VALUES ('refresh_file', 'pkg', 'running', 1, 0, 0, 0)"
        )


def test_expired_lease_is_reclaimed_as_retry(db):
    job_id = dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=1
    )
    claimed = dal.claim_next_job(db, now_ms=1000, lease_ms=5000)
    reclaimed = dal.reclaim_expired_jobs(db, now_ms=10_000)
    assert reclaimed == 1
    job = get_job(db, job_id)
    assert job["status"] == "retry"
    assert job["lease_token"] is None
    # The reclaimed attempt is claimable again.
    again = dal.claim_next_job(db, now_ms=10_001)
    assert again is not None and again["attempt_count"] == 2


def test_unexpired_lease_is_not_reclaimed(db):
    dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=1
    )
    dal.claim_next_job(db, now_ms=1000, lease_ms=60_000)
    assert dal.reclaim_expired_jobs(db, now_ms=2000) == 0


def test_complete_job_requires_matching_lease_and_revision(db):
    job_id = dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=3
    )
    claimed = dal.claim_next_job(db, now_ms=1000)

    # Wrong lease token must be refused.
    with pytest.raises(dal.LeaseMismatchError):
        dal.complete_job(
            db, job_id=job_id, lease_token=new_lease_token(),
            desired_revision=3, now_ms=2000,
        )

    # Wrong revision (source changed mid-flight) must be refused.
    with pytest.raises(dal.StaleJobError):
        dal.complete_job(
            db, job_id=job_id, lease_token=claimed["lease_token"],
            desired_revision=4, now_ms=2000,
        )

    dal.complete_job(
        db, job_id=job_id, lease_token=claimed["lease_token"],
        desired_revision=3, now_ms=2000,
    )
    # The completed job leaves the queue entirely.
    assert db.execute(
        "SELECT COUNT(*) FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()[0] == 0


def test_complete_removes_terminal_job_from_queue(db):
    job_id = dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=1
    )
    claimed = dal.claim_next_job(db, now_ms=1000)
    dal.complete_job(
        db, job_id=job_id, lease_token=claimed["lease_token"],
        desired_revision=1, now_ms=2000,
    )
    assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_fail_job_is_terminal_until_new_revision_requeues(db):
    job_id = dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=1
    )
    claimed = dal.claim_next_job(db, now_ms=1000)
    dal.fail_job(
        db, job_id=job_id, lease_token=claimed["lease_token"],
        last_error="permanent parse error", now_ms=2000,
    )
    job = get_job(db, job_id)
    assert job["status"] == "failed"
    assert job["last_error"] == "permanent parse error"
    # A failed job is not claimable.
    assert dal.claim_next_job(db, now_ms=3000) is None
    # A new source revision requeues it.
    dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=2
    )
    again = dal.claim_next_job(db, now_ms=4000)
    assert again is not None and again["attempt_count"] == 1


def test_fifo_ordering(db):
    first = dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="a.py", desired_revision=1,
        available_at_ms=0,
    )
    second = dal.enqueue_job(
        db, command="refresh_file", folder="pkg", file="b.py", desired_revision=1,
        available_at_ms=1,
    )
    assert dal.claim_next_job(db, now_ms=100)["id"] == first
    assert dal.claim_next_job(db, now_ms=100)["id"] == second
