"""Job processing pipeline (plan §7.2–7.4): claim, stable read, changed-
entity-only inference, atomic publication, backoff, and recovery."""

import json
import sqlite3

import pytest

from indexedcode.dal import (
    claim_next_job,
    enqueue_job,
    reclaim_expired_jobs,
)
from indexedcode.pipeline import (
    MAX_ATTEMPTS,
    process_next_job,
)

from tests.conftest import FakeEmbedder, FakeSummarizer, insert_module

PY_A = """def foo():
    return 1


def bar():
    return 2
"""

PY_A_MOVED = """# a leading comment moves everything down


def foo():
    return 1


def bar():
    return 2
"""

PY_A_BAR_CHANGED = """def foo():
    return 1


def bar():
    return 20
"""

PY_BROKEN = "def broken(:\n    pass\n"


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    return tmp_path


def indexed_file_row(conn, folder, file):
    return conn.execute(
        "SELECT * FROM files WHERE folder = ? AND file = ?",
        (folder, file),
    ).fetchone()


def function_row(conn, file_id, qualified_name):
    return conn.execute(
        "SELECT * FROM functions WHERE file_id = ? AND qualified_name = ?",
        (file_id, qualified_name),
    ).fetchone()


def entity_index_row(conn, entity_type, entity_id):
    return conn.execute(
        "SELECT * FROM entity_index WHERE entity_type = ? AND entity_id = ?",
        (entity_type, entity_id),
    ).fetchone()


def job_metrics(conn):
    return conn.execute(
        "SELECT outcome, queue_duration_ms, process_duration_ms, detail_json,"
        " command, error_code FROM metrics WHERE metric_kind = 'job'"
        " ORDER BY id"
    ).fetchall()


def seed_and_process(conn, repo, source, embed=None, now_ms=1000):
    """Index one python file end-to-end and return the embedder used."""
    (repo / "src").mkdir(exist_ok=True)
    (repo / "src" / "a.py").write_text(source)
    enqueue_job(
        conn,
        command="refresh_file",
        folder="src",
        file="a.py",
        now_ms=now_ms,
    )
    embed = embed if embed is not None else FakeEmbedder()
    outcome = process_next_job(
        conn, str(repo), embed=embed, summarize=FakeSummarizer(),
        now_ms=now_ms + 1,
    )
    assert outcome == "completed"
    return embed


def test_end_to_end_refresh_indexes_entities(db, repo):
    embed = seed_and_process(db, repo, PY_A)

    row = indexed_file_row(db, "src", "a.py")
    assert row["status"] == "ready"
    assert row["indexed_hash"] is not None
    foo = function_row(db, row["id"], "foo")
    bar = function_row(db, row["id"], "bar")
    assert foo is not None and bar is not None
    assert foo["summary"].startswith("summary of")
    index_foo = entity_index_row(db, "function", foo["id"])
    assert index_foo is not None
    assert len(index_foo["embedding"]) == 3 * 4  # float32 x dims
    assert len(embed.calls) == 1  # one batch


def test_location_only_edit_updates_coordinates_without_inference(db, repo):
    embed = seed_and_process(db, repo, PY_A)
    baseline_calls = len(embed.calls)
    old_foo = function_row(db, indexed_file_row(db, "src", "a.py")["id"], "foo")
    old_index = entity_index_row(db, "function", old_foo["id"])

    (repo / "src" / "a.py").write_text(PY_A_MOVED)
    enqueue_job(
        db, command="refresh_file", folder="src", file="a.py", now_ms=2000
    )
    outcome = process_next_job(
        db, str(repo), embed=embed, summarize=FakeSummarizer(), now_ms=2001
    )

    assert outcome == "completed"
    assert len(embed.calls) == baseline_calls  # no inference after the move
    new_row = indexed_file_row(db, "src", "a.py")
    new_foo = function_row(db, new_row["id"], "foo")
    assert new_foo["start_line"] == 4  # coordinates updated (snapshot replaced)
    assert old_foo["start_line"] == 1
    new_index = entity_index_row(db, "function", new_foo["id"])
    assert new_index is not None
    assert new_index["embedding"] == old_index["embedding"]
    assert new_index["input_hash"] == old_index["input_hash"]


def test_changed_entity_is_reembedded_but_unchanged_sibling_is_not(db, repo):
    embed = seed_and_process(db, repo, PY_A)
    baseline_calls = len(embed.calls)

    (repo / "src" / "a.py").write_text(PY_A_BAR_CHANGED)
    enqueue_job(
        db, command="refresh_file", folder="src", file="a.py", now_ms=3000
    )
    outcome = process_next_job(
        db, str(repo), embed=embed, summarize=FakeSummarizer(), now_ms=3001
    )

    assert outcome == "completed"
    assert len(embed.calls) == baseline_calls + 1
    assert len(embed.calls[-1]) == 1  # only bar
    assert "return 20" in embed.calls[-1][0]


def test_removed_entity_index_rows_are_dropped(db, repo):
    seed_and_process(db, repo, PY_A)

    (repo / "src" / "a.py").write_text("def foo():\n    return 1\n")
    enqueue_job(
        db, command="refresh_file", folder="src", file="a.py", now_ms=4000
    )
    process_next_job(
        db, str(repo), embed=FakeEmbedder(), summarize=FakeSummarizer(),
        now_ms=4001,
    )

    file_id = indexed_file_row(db, "src", "a.py")["id"]
    assert function_row(db, file_id, "bar") is None
    remaining = db.execute(
        "SELECT entity_id FROM entity_index ei WHERE entity_type = 'function'"
        " AND entity_id IN (SELECT id FROM functions WHERE file_id = ?)",
        (file_id,),
    ).fetchall()
    assert len(remaining) == 1  # only foo


def test_parse_error_never_replaces_known_good_snapshot(db, repo):
    seed_and_process(db, repo, PY_A)
    good_row = indexed_file_row(db, "src", "a.py")
    good_hash = good_row["indexed_hash"]
    good_foo = function_row(db, good_row["id"], "foo")

    (repo / "src" / "a.py").write_text(PY_BROKEN)
    enqueue_job(
        db, command="refresh_file", folder="src", file="a.py", now_ms=5000
    )
    outcome = process_next_job(
        db, str(repo), embed=FakeEmbedder(), summarize=FakeSummarizer(),
        now_ms=5001,
    )

    assert outcome == "retried"
    row = indexed_file_row(db, "src", "a.py")
    assert row["indexed_hash"] == good_hash  # last-good retained
    assert function_row(db, row["id"], "foo") is not None
    assert good_foo["source_hash"] == function_row(
        db, row["id"], "foo"
    )["source_hash"]


def test_parse_error_on_first_index_never_publishes_partial(db, repo):
    (repo / "src" / "a.py").write_text(PY_BROKEN)
    enqueue_job(
        db, command="refresh_file", folder="src", file="a.py", now_ms=1000
    )
    outcome = process_next_job(
        db, str(repo), embed=FakeEmbedder(), summarize=FakeSummarizer(),
        now_ms=1001,
    )
    assert outcome == "retried"
    row = indexed_file_row(db, "src", "a.py")
    assert row["indexed_hash"] is None
    assert db.execute(
        "SELECT COUNT(*) FROM functions"
    ).fetchone()[0] == 0


def test_transient_inference_failure_preserves_last_good(db, repo):
    seed_and_process(db, repo, PY_A)
    before = indexed_file_row(db, "src", "a.py")

    (repo / "src" / "a.py").write_text(PY_A_BAR_CHANGED)
    enqueue_job(
        db, command="refresh_file", folder="src", file="a.py", now_ms=6000
    )
    outcome = process_next_job(
        db, str(repo),
        embed=FakeEmbedder(error=RuntimeError("worker overloaded")),
        summarize=FakeSummarizer(), now_ms=6001,
    )

    assert outcome == "retried"
    after = indexed_file_row(db, "src", "a.py")
    assert after["indexed_hash"] == before["indexed_hash"]
    assert function_row(db, after["id"], "bar") is not None
    job = db.execute(
        "SELECT * FROM jobs WHERE command = 'refresh_file'"
        " ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert job["status"] == "retry"
    assert job["attempt_count"] == 1
    assert "worker overloaded" in job["last_error"]


def test_retry_respects_backoff_delay_and_growth(db, repo):
    (repo / "src" / "a.py").write_text(PY_BROKEN)
    enqueue_job(
        db, command="refresh_file", folder="src", file="a.py", now_ms=0
    )
    base = 1000
    prev_available = None
    now = 0
    delays = []
    for attempt in range(3):
        outcome = process_next_job(
            db, str(repo), embed=FakeEmbedder(),
            summarize=FakeSummarizer(), now_ms=now, retry_base_ms=base,
            retry_cap_ms=60_000, jitter=lambda: 0,
        )
        assert outcome == "retried"
        job = db.execute(
            "SELECT * FROM jobs ORDER BY id DESC LIMIT 1"
        ).fetchone()
        available = job["available_at_ms"]
        if prev_available is not None:
            delays.append(available - prev_available)
        else:
            assert available >= now + base
        prev_available = available
        now = available  # jump to when it becomes due
    # exponential growth: second retry waits twice as long
    assert delays[0] == 2 * base


def test_retry_backoff_is_capped(db, repo):
    (repo / "src" / "a.py").write_text(PY_BROKEN)
    enqueue_job(
        db, command="refresh_file", folder="src", file="a.py", now_ms=0
    )
    cap = 5000
    now = 0
    for _ in range(6):
        process_next_job(
            db, str(repo), embed=FakeEmbedder(),
            summarize=FakeSummarizer(), now_ms=now, retry_base_ms=1000,
            retry_cap_ms=cap, jitter=lambda: 0,
        )
        job = db.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 1").fetchone()
        assert job["available_at_ms"] - now <= cap
        now = job["available_at_ms"]


def test_permanent_failure_after_max_attempts(db, repo):
    (repo / "src" / "a.py").write_text(PY_BROKEN)
    enqueue_job(
        db, command="refresh_file", folder="src", file="a.py", now_ms=0
    )
    now = 0
    last_outcome = None
    for _ in range(MAX_ATTEMPTS):
        last_outcome = process_next_job(
            db, str(repo), embed=FakeEmbedder(),
            summarize=FakeSummarizer(), now_ms=now, retry_base_ms=1,
            retry_cap_ms=1, jitter=lambda: 0,
        )
        job = db.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 1").fetchone()
        now = max(now, job["available_at_ms"]) + 1
    assert last_outcome == "failed"
    job = db.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 1").fetchone()
    assert job["status"] == "failed"
    assert job["last_error"]


def test_source_changed_during_processing_discards_result(db, repo):
    """Stale in-flight work cannot publish over a newer revision."""
    seed_and_process(db, repo, PY_A)
    (repo / "src" / "a.py").write_text(PY_A_BAR_CHANGED)
    enqueue_job(
        db, command="refresh_file", folder="src", file="a.py", now_ms=9000
    )

    # the embedder is "slow": the file changes again while it runs
    embed = FakeEmbedder()

    def racing_embed(texts):
        (repo / "src" / "a.py").write_text(PY_A)
        return embed(texts)

    outcome = process_next_job(
        db, str(repo), embed=racing_embed, summarize=FakeSummarizer(),
        now_ms=9001,
    )
    assert outcome == "retried"
    # the interim content was never published
    row = indexed_file_row(db, "src", "a.py")
    assert function_row(db, row["id"], "bar")["source"] .count("2") >= 1


def test_remove_file_job_tombstones_and_cleans_descendants(db, repo):
    seed_and_process(db, repo, PY_A)
    enqueue_job(
        db, command="remove_file", folder="src", file="a.py", now_ms=10000
    )
    outcome = process_next_job(
        db, str(repo), embed=FakeEmbedder(), summarize=FakeSummarizer(),
        now_ms=10001,
    )
    assert outcome == "removed"
    row = indexed_file_row(db, "src", "a.py")
    assert row["status"] == "deleted"
    assert db.execute("SELECT COUNT(*) FROM functions").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM entity_index").fetchone()[0] == 0


def test_expired_running_job_is_reclaimed_and_reprocessed(db, repo):
    (repo / "src" / "a.py").write_text(PY_A)
    enqueue_job(
        db, command="refresh_file", folder="src", file="a.py", now_ms=0
    )
    # simulate a crash: claim with a lease, then never finish
    job = claim_next_job(db, now_ms=0, lease_ms=1000)
    assert job is not None
    assert process_next_job(db, str(repo), embed=FakeEmbedder(),
                            summarize=FakeSummarizer(), now_ms=500) is None
    assert reclaim_expired_jobs(db, now_ms=2000) == 1
    outcome = process_next_job(
        db, str(repo), embed=FakeEmbedder(), summarize=FakeSummarizer(),
        now_ms=2001,
    )
    assert outcome == "completed"
    assert indexed_file_row(db, "src", "a.py")["status"] == "ready"


def test_no_job_returns_none(db, repo):
    assert process_next_job(
        db, str(repo), embed=FakeEmbedder(), summarize=FakeSummarizer(),
        now_ms=0,
    ) is None


def test_job_metrics_recorded_with_timing_and_no_leaks(db, repo):
    seed_and_process(db, repo, PY_A, now_ms=1000)
    rows = job_metrics(db)
    assert len(rows) == 1
    success = rows[0]
    assert success["outcome"] == "success"
    assert success["queue_duration_ms"] >= 0
    assert success["process_duration_ms"] >= 0
    # no source or path data anywhere in the metric row
    for value in (success["detail_json"], success["command"],
                  success["error_code"]):
        if value:
            assert "a.py" not in value
            assert "def foo" not in json.dumps(value)

    # a retry then a failure each append a bounded outcome row
    (repo / "src" / "a.py").write_text(PY_BROKEN)
    enqueue_job(
        db, command="refresh_file", folder="src", file="a.py", now_ms=2000
    )
    now = 2000
    for _ in range(MAX_ATTEMPTS):
        outcome = process_next_job(
            db, str(repo), embed=FakeEmbedder(),
            summarize=FakeSummarizer(), now_ms=now, retry_base_ms=1,
            retry_cap_ms=1, jitter=lambda: 0,
        )
        job = db.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT 1").fetchone()
        now = max(now, job["available_at_ms"]) + 1
    outcomes = [row["outcome"] for row in job_metrics(db)]
    assert "success" in outcomes
    assert "retry" in outcomes
    assert "failed" in outcomes
    for row in job_metrics(db):
        blob = json.dumps(
            {"d": row["detail_json"], "c": row["command"],
             "e": row["error_code"]}
        )
        assert "a.py" not in blob
        assert "broken" not in blob


def test_files_row_created_by_pipeline_when_missing(db, repo):
    """A refresh job for a never-scanned file still finds its module."""
    (repo / "src").mkdir(exist_ok=True)
    insert_module(db, name="src", relative_path="src")
    (repo / "src" / "new.py").write_text(PY_A)
    enqueue_job(
        db, command="refresh_file", folder="src", file="new.py", now_ms=0
    )
    outcome = process_next_job(
        db, str(repo), embed=FakeEmbedder(), summarize=FakeSummarizer(),
        now_ms=1,
    )
    assert outcome == "completed"
    assert indexed_file_row(db, "src", "new.py")["status"] == "ready"
