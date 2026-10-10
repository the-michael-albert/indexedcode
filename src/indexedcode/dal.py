"""Data access layer: job queue, polymorphic entity index, metrics, and
atomic file-refresh publication (plan §5, §7.3)."""

from __future__ import annotations

import json
import uuid

from indexedcode.models import (
    ClassDef,
    EntityIndexInput,
    FunctionDef,
    Job,
    NewJob,
    RelationshipRecord,
)

JOB_COMMANDS = ("refresh_file", "remove_file", "refresh_module")
ENTITY_TABLES = {
    "module": "modules",
    "file": "files",
    "class": "classes",
    "function": "functions",
}

# plan §5.4: only low-cardinality numeric/boolean counters are allowed in
# metrics detail_json. The writer rejects everything else.
APPROVED_DETAIL_KEYS = frozenset(
    {
        "entities_changed",
        "inference_calls",
        "stale_discarded",
        "rss_mb",
        "accelerator_allocated_mb",
        "accelerator_reserved_mb",
        "memory_pressure",
        "restart_count",
        "retry_count",
        "rejected_count",
        "deferred_count",
    }
)


class LeaseMismatchError(RuntimeError):
    """The supplied lease token does not own the job."""


class StaleJobError(RuntimeError):
    """The job's desired revision no longer matches the observed source."""


class PolymorphicIntegrityError(RuntimeError):
    """entity_index references an entity that does not exist."""


class MetricValidationError(RuntimeError):
    """metrics detail contains unapproved keys or values."""


# --- jobs -------------------------------------------------------------------


def _job_id_of(job_id):
    return getattr(job_id, "id", job_id)


def enqueue_job(
    conn,
    command=None,
    folder=None,
    file=None,
    function_name=None,
    line_number=None,
    desired_revision=1,
    payload=None,
    available_at_ms=0,
    now_ms=0,
):
    """Create a durable command, coalescing same-path refreshes.

    Accepts either a NewJob model or the equivalent keyword arguments; all
    inputs are validated by the model before any SQL runs. A pending/retry
    job for the same (command, folder, file) is advanced to the latest
    desired revision instead of duplicated. A failed job is requeued with
    its attempt counter reset. A running job is never mutated. Returns the
    resulting Job.
    """
    if isinstance(command, NewJob):
        spec = command
    else:
        spec = NewJob(
            command=command,
            folder=folder,
            file=file,
            function_name=function_name,
            line_number=line_number,
            desired_revision=desired_revision,
            payload=payload or {},
            available_at_ms=available_at_ms,
            now_ms=now_ms,
        )
    command = spec.command.value
    folder = spec.folder
    file = spec.file
    desired_revision = spec.desired_revision

    row = conn.execute(
        "SELECT id, status, desired_revision FROM jobs"
        " WHERE command = ? AND folder = ? AND file IS ? LIMIT 1",
        (command, folder, file),
    ).fetchone()
    if row is not None and row["status"] in ("pending", "retry", "failed"):
        reset = row["status"] == "failed"
        conn.execute(
            "UPDATE jobs SET"
            " desired_revision = ?,"
            " status = 'pending',"
            " attempt_count = CASE WHEN ? THEN 0 ELSE attempt_count END,"
            " last_error = NULL,"
            " available_at_ms = CASE WHEN ? THEN ? ELSE available_at_ms END,"
            " function_name = ?,"
            " line_number = ?,"
            " payload_json = ?,"
            " updated_at_ms = ?"
            " WHERE id = ?",
            (
                desired_revision,
                reset,
                reset,
                spec.available_at_ms,
                spec.function_name,
                spec.line_number,
                json.dumps(spec.payload),
                spec.now_ms,
                row["id"],
            ),
        )
        job_id = row["id"]
    else:
        cur = conn.execute(
            "INSERT INTO jobs (command, folder, file, function_name, line_number,"
            " payload_json, status, desired_revision, available_at_ms,"
            " created_at_ms, updated_at_ms)"
            " VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)",
            (
                command,
                folder,
                file,
                spec.function_name,
                spec.line_number,
                json.dumps(spec.payload),
                desired_revision,
                spec.available_at_ms,
                spec.now_ms,
                spec.now_ms,
            ),
        )
        job_id = cur.lastrowid
    return _fetch_job(conn, job_id)


def _fetch_job(conn, job_id):
    row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return None if row is None else Job.model_validate(dict(row))


def claim_next_job(conn, now_ms, lease_ms=60_000):
    """Atomically claim the earliest due job with a lease token, or None."""
    row = conn.execute(
        "SELECT id FROM jobs"
        " WHERE status IN ('pending', 'retry') AND available_at_ms <= ?"
        " ORDER BY available_at_ms, id LIMIT 1",
        (now_ms,),
    ).fetchone()
    if row is None:
        return None
    token = uuid.uuid4().hex
    conn.execute(
        "UPDATE jobs SET status = 'running', lease_token = ?,"
        " lease_expires_at_ms = ?, attempt_count = attempt_count + 1,"
        " updated_at_ms = ? WHERE id = ?",
        (token, now_ms + lease_ms, now_ms, row["id"]),
    )
    return _fetch_job(conn, row["id"])


def reclaim_expired_jobs(conn, now_ms):
    """Return expired running jobs to the retry queue; returns the count."""
    cur = conn.execute(
        "UPDATE jobs SET status = 'retry', lease_token = NULL,"
        " lease_expires_at_ms = NULL, available_at_ms = ?, updated_at_ms = ?"
        " WHERE status = 'running' AND lease_expires_at_ms <= ?",
        (now_ms, now_ms, now_ms),
    )
    return cur.rowcount


def _require_lease(conn, job_id, lease_token):
    job_id = _job_id_of(job_id)
    row = conn.execute(
        "SELECT lease_token FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()
    if row is None or row["lease_token"] != lease_token:
        raise LeaseMismatchError(f"job {job_id} is not owned by this lease")


def retry_job(conn, job_id, lease_token, last_error, retry_at_ms, now_ms):
    """Release the lease and reschedule the job with capped backoff."""
    _require_lease(conn, job_id, lease_token)
    conn.execute(
        "UPDATE jobs SET status = 'retry', lease_token = NULL,"
        " lease_expires_at_ms = NULL, last_error = ?, available_at_ms = ?,"
        " updated_at_ms = ? WHERE id = ?",
        (last_error, retry_at_ms, now_ms, _job_id_of(job_id)),
    )


def fail_job(conn, job_id, lease_token, last_error, now_ms):
    """Mark the job permanently failed; a new revision requeues it."""
    _require_lease(conn, job_id, lease_token)
    conn.execute(
        "UPDATE jobs SET status = 'failed', lease_token = NULL,"
        " lease_expires_at_ms = NULL, last_error = ?, updated_at_ms = ?"
        " WHERE id = ?",
        (last_error, now_ms, _job_id_of(job_id)),
    )


def complete_job(conn, job_id, lease_token, desired_revision, now_ms):
    """Complete the job under its lease; verifies revision first.

    Terminal jobs are retained with status 'done' so their job-attempt
    metrics keep a valid FK reference (plan §5.4). Claims only consider
    pending/retry rows, so done jobs never rerun.
    """
    row = conn.execute(
        "SELECT lease_token, desired_revision FROM jobs WHERE id = ?",
        (_job_id_of(job_id),),
    ).fetchone()
    if row is None or row["lease_token"] != lease_token:
        raise LeaseMismatchError(f"job {job_id} is not owned by this lease")
    if row["desired_revision"] != desired_revision:
        raise StaleJobError(
            f"job {job_id} targets revision {row['desired_revision']},"
            f" not {desired_revision}"
        )
    conn.execute(
        "UPDATE jobs SET status = 'done', lease_token = NULL,"
        " lease_expires_at_ms = NULL, last_error = NULL, updated_at_ms = ?"
        " WHERE id = ?",
        (now_ms, _job_id_of(job_id)),
    )


# --- entity_index ------------------------------------------------------------


def _validate_entity_ref(conn, entity_type, entity_id):
    table = ENTITY_TABLES.get(entity_type)
    if table is None:
        raise ValueError(f"unknown entity_type: {entity_type!r}")
    row = conn.execute(
        f"SELECT id FROM {table} WHERE id = ?", (entity_id,)
    ).fetchone()
    if row is None:
        raise PolymorphicIntegrityError(
            f"{entity_type} #{entity_id} does not exist"
        )


def _entity_index_input(entity_type, entity_id, **kwargs):
    if isinstance(entity_type, EntityIndexInput):
        return entity_type
    return EntityIndexInput(
        entity_type=entity_type, entity_id=entity_id, **kwargs
    )


def insert_entity_index(
    conn,
    entity_type,
    entity_id=None,
    search_text=None,
    embedding=None,
    dimensions=None,
    input_hash=None,
    model_fingerprint=None,
    embedding_truncated=False,
    now_ms=0,
):
    """Insert one search row, validating the polymorphic target exists.

    Accepts an EntityIndexInput model or the equivalent keyword arguments.
    """
    inp = _entity_index_input(
        entity_type,
        entity_id,
        search_text=search_text,
        embedding=embedding,
        dimensions=dimensions,
        input_hash=input_hash,
        model_fingerprint=model_fingerprint,
        embedding_truncated=embedding_truncated,
        now_ms=now_ms,
    )
    _validate_entity_ref(conn, inp.entity_type.value, inp.entity_id)
    conn.execute(
        "INSERT INTO entity_index (entity_type, entity_id, search_text,"
        " embedding, embedding_dimensions, input_hash, model_fingerprint,"
        " embedding_truncated, updated_at_ms)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            inp.entity_type.value,
            inp.entity_id,
            inp.search_text,
            inp.embedding,
            inp.dimensions,
            inp.input_hash,
            inp.model_fingerprint,
            1 if inp.embedding_truncated else 0,
            inp.now_ms,
        ),
    )


def upsert_entity_index(
    conn,
    entity_type,
    entity_id=None,
    search_text=None,
    embedding=None,
    dimensions=None,
    input_hash=None,
    model_fingerprint=None,
    embedding_truncated=False,
    now_ms=0,
):
    """Replace the search row for an entity after a re-embedding."""
    inp = _entity_index_input(
        entity_type,
        entity_id,
        search_text=search_text,
        embedding=embedding,
        dimensions=dimensions,
        input_hash=input_hash,
        model_fingerprint=model_fingerprint,
        embedding_truncated=embedding_truncated,
        now_ms=now_ms,
    )
    _validate_entity_ref(conn, inp.entity_type.value, inp.entity_id)
    conn.execute(
        "INSERT INTO entity_index (entity_type, entity_id, search_text,"
        " embedding, embedding_dimensions, input_hash, model_fingerprint,"
        " embedding_truncated, updated_at_ms)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        " ON CONFLICT (entity_type, entity_id) DO UPDATE SET"
        " search_text = excluded.search_text,"
        " embedding = excluded.embedding,"
        " embedding_dimensions = excluded.embedding_dimensions,"
        " input_hash = excluded.input_hash,"
        " model_fingerprint = excluded.model_fingerprint,"
        " embedding_truncated = excluded.embedding_truncated,"
        " updated_at_ms = excluded.updated_at_ms",
        (
            inp.entity_type.value,
            inp.entity_id,
            inp.search_text,
            inp.embedding,
            inp.dimensions,
            inp.input_hash,
            inp.model_fingerprint,
            1 if inp.embedding_truncated else 0,
            inp.now_ms,
        ),
    )


def delete_entity_index_for_file(conn, file_id):
    """Delete all search rows belonging to a file's entities (and the file)."""
    cur = conn.execute(
        "DELETE FROM entity_index WHERE"
        " (entity_type = 'function' AND entity_id IN"
        "   (SELECT id FROM functions WHERE file_id = ?))"
        " OR (entity_type = 'class' AND entity_id IN"
        "   (SELECT id FROM classes WHERE file_id = ?))"
        " OR (entity_type = 'file' AND entity_id = ?)",
        (file_id, file_id, file_id),
    )
    return cur.rowcount


# --- metrics -----------------------------------------------------------------


def _validated_detail(detail):
    if not detail:
        return {}
    for key, value in detail.items():
        if key not in APPROVED_DETAIL_KEYS:
            raise MetricValidationError(f"unapproved metrics detail key: {key!r}")
        if not isinstance(value, (int, float, bool)):
            raise MetricValidationError(
                f"metrics detail value for {key!r} must be numeric or boolean"
            )
    return dict(detail)


def record_metric(
    conn,
    metric_kind,
    operation,
    outcome,
    component,
    now_ms,
    detail=None,
    request_id=None,
    status_code=None,
    request_duration_ms=None,
    request_bytes=None,
    response_bytes=None,
    job_id=None,
    job_attempt=None,
    queue_duration_ms=None,
    process_duration_ms=None,
    command=None,
    backend=None,
    error_code=None,
):
    """Append one bounded observability row (plan §5.4).

    Missing required per-kind fields are rejected by the schema CHECKs and
    surface as sqlite3.IntegrityError.
    """
    detail_json = json.dumps(_validated_detail(detail))
    conn.execute(
        "INSERT INTO metrics (recorded_at_ms, metric_kind, operation, outcome,"
        " component, request_id, status_code, request_duration_ms,"
        " request_bytes, response_bytes, job_id, job_attempt,"
        " queue_duration_ms, process_duration_ms, command, backend,"
        " error_code, detail_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            now_ms,
            metric_kind,
            operation,
            outcome,
            component,
            request_id,
            status_code,
            request_duration_ms,
            request_bytes,
            response_bytes,
            job_id,
            job_attempt,
            queue_duration_ms,
            process_duration_ms,
            command,
            backend,
            error_code,
            detail_json,
        ),
    )


def prune_metrics(conn, now_ms, max_age_ms, batch=1000):
    """Delete metrics older than the retention age in bounded batches."""
    cur = conn.execute(
        "DELETE FROM metrics WHERE id IN ("
        "  SELECT id FROM metrics WHERE recorded_at_ms < ?"
        "  ORDER BY recorded_at_ms LIMIT ?)",
        (now_ms - max_age_ms, batch),
    )
    return cur.rowcount


# --- atomic file publication (plan §7.3) --------------------------------------


def _insert_class_row(conn, file_id, cls, now_ms):
    conn.execute(
        "INSERT INTO classes (file_id, parent_class_id, name, qualified_name,"
        " kind, start_line, end_line, start_byte, end_byte, source,"
        " source_hash, updated_at_ms)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            file_id,
            cls["parent_class_id"],
            cls["name"],
            cls["qualified_name"],
            cls["kind"],
            cls["start_line"],
            cls["end_line"],
            cls["start_byte"],
            cls["end_byte"],
            cls["source"],
            cls["source_hash"],
            now_ms,
        ),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def _insert_function_row(conn, file_id, fn, now_ms):
    conn.execute(
        "INSERT INTO functions (file_id, class_id, parent_function_id, name,"
        " qualified_name, signature, start_line, end_line, start_byte,"
        " end_byte, source, source_hash, updated_at_ms)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            file_id,
            fn["class_id"],
            fn["parent_function_id"],
            fn["name"],
            fn["qualified_name"],
            fn["signature"],
            fn["start_line"],
            fn["end_line"],
            fn["start_byte"],
            fn["end_byte"],
            fn["source"],
            fn["source_hash"],
            now_ms,
        ),
    )
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def publish_file_refresh(conn, job, file_row, classes, functions, relationships,
                         now_ms, indexed_hash=None):
    """Replace a file's snapshot atomically, then complete its job.

    Accepts ClassDef/FunctionDef/RelationshipRecord models or the equivalent
    dicts; all entities are validated by their models before any SQL runs.
    Either the previous complete snapshot or the next one is visible to
    readers. Metrics are appended best-effort after the commit and can never
    roll back a successful publication.
    """
    classes = [c if isinstance(c, ClassDef) else ClassDef.model_validate(c)
               for c in classes]
    functions = [f if isinstance(f, FunctionDef) else FunctionDef.model_validate(f)
                 for f in functions]
    relationships = [
        r if isinstance(r, RelationshipRecord)
        else RelationshipRecord.model_validate(r)
        for r in relationships
    ]
    conn.execute("BEGIN IMMEDIATE")
    try:
        file_id = conn.execute(
            "SELECT id FROM files WHERE folder = ? AND file = ?",
            (file_row["folder"], file_row["file"]),
        ).fetchone()[0]

        # 1. remove the previous snapshot
        conn.execute(
            "DELETE FROM relationships WHERE origin_file_id = ?", (file_id,)
        )
        delete_entity_index_for_file(conn, file_id)
        conn.execute("DELETE FROM functions WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM classes WHERE file_id = ?", (file_id,))

        # 2-3. insert refreshed entities and relationships
        for cls in classes:
            _insert_class_row(conn, file_id, cls, now_ms)
        for fn in functions:
            _insert_function_row(conn, file_id, fn, now_ms)
        for rel in relationships:
            conn.execute(
                "INSERT INTO relationships (source_type, source_id, relation,"
                " target_type, target_id, target_name, origin_file_id,"
                " line_number, resolution, snippet)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rel["source_type"],
                    rel["source_id"],
                    rel["relation"],
                    rel["target_type"],
                    rel["target_id"],
                    rel["target_name"],
                    file_id,
                    rel["line_number"],
                    rel["resolution"],
                    rel["snippet"],
                ),
            )

        # 5. mark the file ready and complete the job under its lease
        conn.execute(
            "UPDATE files SET status = 'ready', indexed_hash = ?,"
            " indexed_revision = ?, last_error = NULL, updated_at_ms = ?"
            " WHERE id = ?",
            (indexed_hash, job["desired_revision"], now_ms, file_id),
        )
        complete_job(
            conn,
            job_id=job["id"],
            lease_token=job["lease_token"],
            desired_revision=job["desired_revision"],
            now_ms=now_ms,
        )
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise

    # 6. best-effort terminal job metric; never affects the committed result
    try:
        record_metric(
            conn,
            metric_kind="job",
            operation=job["command"],
            outcome="success",
            component="orchestrator",
            now_ms=now_ms,
            job_id=job["id"],
            job_attempt=job["attempt_count"],
            queue_duration_ms=0,
            process_duration_ms=0,
            command=job["command"],
        )
    except Exception:
        pass  # bounded internal counter; plan §5.4


def remove_file(conn, folder, file, now_ms):
    """Tombstone a removed file and delete its descendants in one transaction."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT id FROM files WHERE folder = ? AND file = ?", (folder, file)
        ).fetchone()
        if row is None:
            conn.execute("ROLLBACK")
            return None
        file_id = row[0]
        conn.execute(
            "DELETE FROM relationships WHERE origin_file_id = ?", (file_id,)
        )
        delete_entity_index_for_file(conn, file_id)
        conn.execute("DELETE FROM functions WHERE file_id = ?", (file_id,))
        conn.execute("DELETE FROM classes WHERE file_id = ?", (file_id,))
        conn.execute(
            "UPDATE files SET status = 'deleted', indexed_hash = NULL,"
            " indexed_revision = NULL, updated_at_ms = ? WHERE id = ?",
            (now_ms, file_id),
        )
        conn.execute("COMMIT")
        return file_id
    except Exception:
        conn.execute("ROLLBACK")
        raise
