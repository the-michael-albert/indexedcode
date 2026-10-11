"""Reconciliation (plan §7.1, §7.4): initial scan, cheap filters, and
authoritative removal.

Candidate files are compared against the store using mtime/size as a
cheap filter and content hash as the authoritative one; unchanged files
never enqueue or re-infer. Deletions are inferred only from a complete
authoritative scan — paths that were merely skipped (unreadable,
oversized, ignored, unsupported) still exist and are never treated as
deleted. A permission error never means deleted.
"""

from __future__ import annotations

import logging
import os

from indexedcode.dal import enqueue_job, ensure_file_row
from indexedcode.scan import scan_repository
from indexedcode.stable import read_stable


logger = logging.getLogger(__name__)


class ReconcileResult:
    def __init__(self):
        self.refresh_enqueued = 0
        self.remove_enqueued = 0
        self.unchanged = 0
        self.skipped = 0
        self.candidates = 0


def reconcile(conn, root, now_ms=0, model_fingerprint=None):
    """Bring the store's pending work in line with the on-disk tree.

    With ``model_fingerprint`` set, ready files whose entities lack a
    vector from that model are re-queued too (embedding backfill).
    """
    result = ReconcileResult()
    scan = scan_repository(root)

    result.candidates = len(scan.files)
    seen = set()
    for skipped in scan.skipped:
        seen.add(skipped.relative_path)
        result.skipped += 1
    for candidate in scan.files:
        seen.add(candidate.relative_path)

    for candidate in scan.files:
        rel = candidate.relative_path
        folder, _, file = rel.rpartition("/")
        path = os.path.join(root, rel)
        try:
            stat = os.stat(path)
        except OSError:
            result.skipped += 1
            continue

        row = conn.execute(
            "SELECT * FROM files WHERE folder = ? AND file = ?",
            (folder, file),
        ).fetchone()

        if (row is not None
                and _has_pending_job(conn, folder, file, "refresh_file")):
            # an unprocessed command already covers this path; refresh
            # only the cheap observation fields so the filter stays sharp
            _observe(conn, row, stat, row["observed_hash"], now_ms)
            continue

        if (row is not None and row["status"] == "error"
                and row["last_error"] == "parse_error"
                and row["observed_mtime_ns"] == stat.st_mtime_ns
                and row["observed_size_bytes"] == stat.st_size):
            # a syntax error is deterministic: retrying the same bytes on
            # every scan only burns CPU. The file is re-queued as soon as it
            # changes on disk (mtime/size differ).
            result.unchanged += 1
            continue

        if row is not None and row["status"] == "ready":
            cheap_match = (
                row["indexed_hash"] is not None
                and row["observed_mtime_ns"] == stat.st_mtime_ns
                and row["observed_size_bytes"] == stat.st_size
            )
            if cheap_match:
                if _needs_backfill(conn, row, model_fingerprint):
                    _enqueue_refresh(
                        conn, row, stat, row["indexed_hash"], now_ms,
                        candidate.language, candidate.is_test)
                    result.refresh_enqueued += 1
                    continue
                result.unchanged += 1
                continue
            try:
                _, digest = read_stable(path)
            except OSError:
                result.skipped += 1
                continue
            if digest == row["indexed_hash"]:
                if _needs_backfill(conn, row, model_fingerprint):
                    _enqueue_refresh(conn, row, stat, digest, now_ms,
                                     candidate.language, candidate.is_test)
                    result.refresh_enqueued += 1
                    continue
                _observe(conn, row, stat, digest, now_ms)
                result.unchanged += 1
                continue
            _enqueue_refresh(conn, row, stat, digest, now_ms,
                             candidate.language, candidate.is_test)
            result.refresh_enqueued += 1
        else:
            # never indexed (or tombstoned): create/refresh the row and
            # enqueue an initial refresh
            if row is None:
                row = ensure_file_row(
                    conn, folder, file, language=candidate.language,
                    is_test=candidate.is_test, now_ms=now_ms,
                )
            _, digest = read_stable(path)
            _enqueue_refresh(conn, row, stat, digest, now_ms,
                             candidate.language, candidate.is_test)
            result.refresh_enqueued += 1

    _enqueue_removals(conn, seen, result, now_ms)
    return result


def _needs_backfill(conn, row, model_fingerprint):
    """True when a ready file has an entity without a current-model vector."""
    if model_fingerprint is None:
        return False
    for table, kind in (("functions", "function"), ("classes", "class")):
        if conn.execute(
            f"SELECT 1 FROM {table} t WHERE t.file_id = ? AND NOT EXISTS ("
            " SELECT 1 FROM entity_index e WHERE e.entity_type = ?"
            " AND e.entity_id = t.id AND e.model_fingerprint = ?) LIMIT 1",
            (row["id"], kind, model_fingerprint),
        ).fetchone() is not None:
            return True
    return False


def _observe(conn, row, stat, digest, now_ms):
    conn.execute(
        "UPDATE files SET observed_mtime_ns = ?, observed_size_bytes = ?,"
        " observed_hash = ?, updated_at_ms = ? WHERE id = ?",
        (stat.st_mtime_ns, stat.st_size, digest, now_ms, row["id"]),
    )


def _enqueue_refresh(conn, row, stat, digest, now_ms, language, is_test):
    if _has_pending_job(conn, row["folder"], row["file"], "refresh_file"):
        # an unprocessed command already covers this path
        _observe(conn, row, stat, digest, now_ms)
        return
    revision = row["observed_revision"] + 1
    conn.execute(
        "UPDATE files SET status = 'pending', language = ?, is_test = ?,"
        " observed_revision = ?, observed_mtime_ns = ?,"
        " observed_size_bytes = ?, observed_hash = ?, updated_at_ms = ?"
        " WHERE id = ?",
        (language, int(bool(is_test)), revision, stat.st_mtime_ns,
         stat.st_size, digest, now_ms, row["id"]),
    )
    job = enqueue_job(
        conn,
        command="refresh_file",
        folder=row["folder"],
        file=row["file"],
        desired_revision=revision,
        now_ms=now_ms,
    )
    rel = (f"{row['folder']}/{row['file']}" if row["folder"]
           else row["file"])
    logger.info("QUEUED   #%d refresh_file %s", job.id, rel)


def _enqueue_removals(conn, seen, result, now_ms):
    rows = conn.execute(
        "SELECT folder, file FROM files WHERE status != 'deleted'"
    ).fetchall()
    for row in rows:
        rel = f"{row['folder']}/{row['file']}" if row["folder"] else row["file"]
        if rel in seen:
            continue
        if _has_pending_job(conn, row["folder"], row["file"], "remove_file"):
            continue
        job = enqueue_job(
            conn,
            command="remove_file",
            folder=row["folder"],
            file=row["file"],
            now_ms=now_ms,
        )
        logger.info("QUEUED   #%d remove_file %s", job.id, rel)
        result.remove_enqueued += 1


def _has_pending_job(conn, folder, file, command):
    return conn.execute(
        "SELECT 1 FROM jobs WHERE folder = ? AND file = ? AND command = ?"
        " AND status IN ('pending', 'retry') LIMIT 1",
        (folder, file, command),
    ).fetchone() is not None
