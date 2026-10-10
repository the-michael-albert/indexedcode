"""Index health status (plan §8.1 code_status)."""

from indexedcode.retrieval import get_status


def test_status_reports_coverage_and_zero_pending(indexed_repo):
    conn, root = indexed_repo
    status = get_status(conn, str(root))
    assert status["coverage"]["ready_files"] == 3
    assert status["coverage"]["known_files"] == 3
    assert status["pending_jobs"] == 0
    assert status["failures"]["failed_jobs"] == 0


def test_status_lists_stale_paths_after_disk_edit(indexed_repo):
    conn, root = indexed_repo
    (root / "src" / "config_loader.py").write_text(
        "def parse_config(text):\n    return {}\n"
    )
    status = get_status(conn, str(root))
    assert "src/config_loader.py" in status["stale_paths"]


def test_status_counts_pending_jobs(indexed_repo):
    conn, root = indexed_repo
    from indexedcode.dal import enqueue_job

    enqueue_job(conn, command="refresh_file", folder="src",
                file="reporting.py", now_ms=9999)
    status = get_status(conn, str(root))
    assert status["pending_jobs"] == 1


def test_status_counts_failures(indexed_repo):
    conn, root = indexed_repo
    from indexedcode.dal import enqueue_job, fail_job, claim_next_job

    enqueue_job(conn, command="refresh_file", folder="src",
                file="reporting.py", now_ms=0)
    job = claim_next_job(conn, now_ms=0)
    fail_job(conn, job.id, job.lease_token, "boom", now_ms=1)
    status = get_status(conn, str(root))
    assert status["failures"]["failed_jobs"] == 1


def test_status_reports_model_fingerprint(indexed_repo):
    conn, root = indexed_repo
    status = get_status(conn, str(root))
    assert status["model_fingerprint"]  # non-empty when embeddings exist
