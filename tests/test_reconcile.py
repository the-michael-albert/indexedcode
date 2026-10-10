"""Reconciliation: initial scan, cheap filters, authoritative removal,
and no re-inference of unchanged files (plan §7.1, §7.4)."""

import pytest

from indexedcode.dal import enqueue_job
from indexedcode.reconcile import reconcile

from tests.conftest import FakeEmbedder, FakeSummarizer


def run_until_idle(conn, root, embed, now_ms=0, step=1000, limit=50):
    """Process every due job, advancing the clock past retry delays."""
    from indexedcode.pipeline import process_next_job

    now = now_ms
    for _ in range(limit):
        outcome = process_next_job(
            conn, str(root), embed=embed, summarize=FakeSummarizer(),
            now_ms=now, retry_base_ms=1, retry_cap_ms=1, jitter=lambda: 0,
        )
        if outcome is None:
            job = conn.execute(
                "SELECT MIN(available_at_ms) AS next FROM jobs"
                " WHERE status IN ('pending', 'retry')"
            ).fetchone()["next"]
            if job is None:
                return now
            now = max(now + step, job + 1)
        else:
            now += step
    raise AssertionError("pipeline never went idle")


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / ".gitignore").write_text("ignored/\n*.log\n")
    return tmp_path


def write(repo, rel, text):
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


PY = "def foo():\n    return 1\n"
PY2 = "def foo():\n    return 2\n"


def test_initial_scan_enqueues_and_indexes_everything(db, repo):
    write(repo, "src/a.py", PY)
    write(repo, "src/b.py", PY2)
    write(repo, "src/ignored/x.py", PY)  # gitignored: skipped
    write(repo, "src/data.bin", b"\x00\x01")  # binary: skipped
    write(repo, "README.md", "hello")  # unsupported: skipped

    result = reconcile(db, str(repo), now_ms=0)

    assert result.refresh_enqueued == 2
    assert result.remove_enqueued == 0
    assert result.skipped >= 3
    run_until_idle(db, repo, FakeEmbedder())
    ready = db.execute(
        "SELECT file FROM files WHERE status = 'ready' ORDER BY file"
    ).fetchall()
    assert [r["file"] for r in ready] == ["a.py", "b.py"]


def test_unchanged_files_do_not_enqueue_or_infer_again(db, repo):
    write(repo, "src/a.py", PY)
    reconcile(db, str(repo), now_ms=0)
    embed = FakeEmbedder()
    run_until_idle(db, repo, embed)
    assert len(embed.calls) == 1

    result = reconcile(db, str(repo), now_ms=10_000)
    assert result.refresh_enqueued == 0
    assert result.unchanged == 1
    run_until_idle(db, repo, embed)
    assert len(embed.calls) == 1  # nothing re-inferred


def test_mtime_touch_without_content_change_is_unchanged(db, repo):
    path = write(repo, "src/a.py", PY)
    reconcile(db, str(repo), now_ms=0)
    run_until_idle(db, repo, FakeEmbedder())

    import os

    os.utime(path, ns=(1, 1))  # old mtime, same content

    result = reconcile(db, str(repo), now_ms=20_000)
    assert result.refresh_enqueued == 0
    assert result.unchanged == 1


def test_changed_file_is_reenqueued_with_bumped_revision(db, repo):
    write(repo, "src/a.py", PY)
    reconcile(db, str(repo), now_ms=0)
    run_until_idle(db, repo, FakeEmbedder())
    old_rev = db.execute(
        "SELECT indexed_revision FROM files WHERE file = 'a.py'"
    ).fetchone()["indexed_revision"]

    write(repo, "src/a.py", PY2)
    result = reconcile(db, str(repo), now_ms=30_000)

    assert result.refresh_enqueued == 1
    run_until_idle(db, repo, FakeEmbedder(), now_ms=30_001)
    row = db.execute("SELECT * FROM files WHERE file = 'a.py'").fetchone()
    assert row["indexed_revision"] == old_rev + 1
    assert db.execute(
        "SELECT source FROM functions WHERE qualified_name = 'foo'"
    ).fetchone()[0].count("2") >= 1


def test_deleted_file_is_removed_only_by_authoritative_scan(db, repo):
    path = write(repo, "src/a.py", PY)
    reconcile(db, str(repo), now_ms=0)
    run_until_idle(db, repo, FakeEmbedder())

    # a permission error never means deleted: unreadable-but-present stays
    path.chmod(0o000)
    result = reconcile(db, str(repo), now_ms=40_000)
    path.chmod(0o644)
    assert result.remove_enqueued == 0
    assert db.execute(
        "SELECT status FROM files WHERE file = 'a.py'"
    ).fetchone()[0] == "ready"

    # real deletion from an authoritative scan enqueues removal
    path.unlink()
    result = reconcile(db, str(repo), now_ms=50_000)
    assert result.remove_enqueued == 1
    run_until_idle(db, repo, FakeEmbedder(), now_ms=50_001)
    row = db.execute("SELECT * FROM files WHERE file = 'a.py'").fetchone()
    assert row["status"] == "deleted"
    assert db.execute("SELECT COUNT(*) FROM functions").fetchone()[0] == 0


def test_rename_is_remove_plus_add(db, repo):
    write(repo, "src/a.py", PY)
    reconcile(db, str(repo), now_ms=0)
    run_until_idle(db, repo, FakeEmbedder())

    (repo / "src" / "a.py").rename(repo / "src" / "renamed.py")
    result = reconcile(db, str(repo), now_ms=60_000)
    assert result.refresh_enqueued == 1
    assert result.remove_enqueued == 1
    run_until_idle(db, repo, FakeEmbedder(), now_ms=60_001)
    statuses = dict(db.execute(
        "SELECT file, status FROM files"
    ).fetchall())
    assert statuses["a.py"] == "deleted"
    assert statuses["renamed.py"] == "ready"


def test_tombstoned_file_reappearance_updates_it(db, repo):
    path = write(repo, "src/a.py", PY)
    reconcile(db, str(repo), now_ms=0)
    run_until_idle(db, repo, FakeEmbedder())
    path.unlink()
    reconcile(db, str(repo), now_ms=70_000)
    run_until_idle(db, repo, FakeEmbedder(), now_ms=70_001)

    write(repo, "src/a.py", PY2)
    result = reconcile(db, str(repo), now_ms=80_000)
    assert result.refresh_enqueued == 1
    run_until_idle(db, repo, FakeEmbedder(), now_ms=80_001)
    row = db.execute("SELECT * FROM files WHERE file = 'a.py'").fetchone()
    assert row["status"] == "ready"
    assert db.execute(
        "SELECT source FROM functions WHERE qualified_name = 'foo'"
    ).fetchone()[0].count("2") >= 1


def test_reconcile_is_idempotent_for_pending_jobs(db, repo):
    write(repo, "src/a.py", PY)
    reconcile(db, str(repo), now_ms=0)
    # second reconcile before processing must not double-enqueue
    result = reconcile(db, str(repo), now_ms=1)
    assert result.refresh_enqueued == 0
    pending = db.execute(
        "SELECT COUNT(*) FROM jobs WHERE status IN ('pending', 'retry')"
    ).fetchone()[0]
    assert pending == 1


def test_previously_skipped_reasons_are_not_reenqueued(db, repo):
    path = write(repo, "src/a.py", PY)
    reconcile(db, str(repo), now_ms=0)
    run_until_idle(db, repo, FakeEmbedder())
    # the file becomes oversized after being indexed
    path.write_text("x = 1  # " + "y" * 2_000_000)
    result = reconcile(db, str(repo), now_ms=90_000)
    # oversized files are skipped observations, not silent refreshes
    assert result.skipped >= 1
