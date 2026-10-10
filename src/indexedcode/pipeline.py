"""Claim-and-process pipeline (plan §7.2–7.4).

One call to :func:`process_next_job` claims the earliest due job under a
lease, takes a stable snapshot of the file, extracts entities, performs
inference only for entities whose source actually changed (outside any
SQLite write transaction), revalidates the snapshot, and publishes
atomically. Parse failures never replace a known-good snapshot; transient
failures retry with capped exponential backoff and jitter; exhausted
attempts fail terminally with a categorical error code.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from indexedcode.dal import (
    claim_next_job,
    complete_job,
    ensure_file_row,
    fail_job,
    publish_file_refresh,
    record_metric,
    remove_file,
    retry_job,
)
from indexedcode.embeddings import pack
from indexedcode.extract import detect_language, extract
from indexedcode.models import (
    ClassDef,
    EntityType,
    FunctionDef,
    RelationshipRecord,
)
from indexedcode.stable import UnstableReadError, read_stable

MAX_ATTEMPTS = 5

_PROCESS_COMMANDS = ("refresh_file", "remove_file")


def process_next_job(
    conn,
    root,
    *,
    embed=None,
    summarize=None,
    now_ms=0,
    lease_ms=60_000,
    retry_base_ms=1000,
    retry_cap_ms=60_000,
    jitter=None,
    model_fingerprint="unpinned",
):
    """Claim and run the next due job; return the outcome string or None.

    Outcomes: ``"completed"``, ``"removed"``, ``"retried"``, ``"failed"``.
    ``embed`` maps a list of texts to a list of float vectors; ``summarize``
    maps a list of texts to a list of summary strings.
    """
    job = claim_next_job(conn, now_ms=now_ms, lease_ms=lease_ms)
    if job is None:
        return None

    started = now_ms
    queue_duration = max(0, started - job["created_at_ms"])

    def finish_retry(code, message, detail=None):
        if job["attempt_count"] >= MAX_ATTEMPTS:
            fail_job(conn, job.id, job.lease_token, f"{code}: {message}",
                     now_ms)
            _record(conn, job, "failed", code, queue_duration,
                    started, now_ms, detail)
            return "failed"
        delay = min(
            retry_cap_ms,
            retry_base_ms * 2 ** (job["attempt_count"] - 1),
        )
        if jitter is not None:
            delay += max(0, jitter())
        retry_job(conn, job.id, job.lease_token, f"{code}: {message}",
                  now_ms + delay, now_ms)
        _record(conn, job, "retry", code, queue_duration,
                started, now_ms, detail)
        return "retried"

    if job.command not in _PROCESS_COMMANDS:
        fail_job(conn, job.id, job.lease_token,
                 f"unsupported command: {job.command}", now_ms)
        _record(conn, job, "failed", "unsupported_command", queue_duration,
                started, now_ms)
        return "failed"

    folder, file = job.folder, job.file
    rel = f"{folder}/{file}" if folder else file
    path = Path(root) / rel

    if job.command == "remove_file":
        remove_file(conn, folder, file, now_ms)
        complete_job(conn, job.id, job.lease_token, job.desired_revision,
                     now_ms)
        _record(conn, job, "success", None, queue_duration, started, now_ms)
        return "removed"

    # --- refresh_file -----------------------------------------------------
    try:
        data, digest = read_stable(path)
    except FileNotFoundError:
        return finish_retry(
            "file_missing", "path vanished; awaiting authoritative scan"
        )
    except UnstableReadError:
        return finish_retry("unstable_read", "file changed during read")

    language = detect_language(file)
    if language is None:
        return finish_retry(
            "unsupported_language", "no extractor for this path"
        )

    extracted = extract(data, language, path=rel)

    if not extracted.parse_ok:
        # never publish a partial parse over a known-good snapshot
        _mark_file_error(conn, folder, file, "parse_error", now_ms,
                         language=language)
        return finish_retry(
            "parse_error", "tree-sitter reported syntax errors"
        )

    file_row = ensure_file_row(
        conn, folder, file, language=language,
        is_test=_is_test(rel), now_ms=now_ms,
    )

    # --- changed-entity detection (§7.2.4: inference only for changes) ----
    old_hashes, old_index = _existing_snapshot(conn, file_row)
    changed_functions = [
        fn for fn in extracted.functions
        if old_hashes.get(("function", fn.qualified_name)) != fn.source_hash
    ]
    changed_classes = [
        cls for cls in extracted.classes
        if old_hashes.get(("class", cls.qualified_name)) != cls.source_hash
    ]
    inference_texts = [fn.source for fn in changed_functions]
    inference_texts += [cls.source for cls in changed_classes]

    vectors: dict[str, list[float]] = {}
    summaries: dict[str, str] = {}
    inference_calls = 0
    if inference_texts:
        try:
            if embed is not None:
                embedded = embed(inference_texts)
                for text, vector in zip(inference_texts, embedded):
                    vectors[text] = vector
                inference_calls += 1
            if summarize is not None:
                for text, summary in zip(inference_texts, summarize(inference_texts)):
                    summaries[text] = summary
                inference_calls += 1
        except Exception as exc:
            _mark_file_error(conn, folder, file, "inference_error", now_ms,
                             language=language)
            return finish_retry(
                "inference_error", f"inference dependency failed: {exc}",
                detail={"inference_calls": inference_calls},
            )

    # --- revalidate the snapshot before publication (§7.2.5) --------------
    try:
        _, digest_after = read_stable(path)
    except (FileNotFoundError, UnstableReadError):
        return finish_retry("source_changed", "snapshot became unreadable")
    if digest_after != digest:
        return finish_retry(
            "source_changed", "source changed while processing",
            detail={"stale_discarded": 1},
        )

    # --- build the publication payload ------------------------------------
    summary_digest = hashlib.sha256(
        (model_fingerprint + "\x00").encode("utf-8")
    ).hexdigest()

    functions = [
        FunctionDef(
            name=fn.name,
            qualified_name=fn.qualified_name,
            signature=fn.signature,
            start_line=fn.start_line,
            end_line=fn.end_line,
            start_byte=fn.start_byte,
            end_byte=fn.end_byte,
            source=fn.source,
            source_hash=fn.source_hash,
            parent_class_qname=fn.parent_class_qname,
            parent_function_qname=fn.parent_function_qname,
            summary=summaries.get(fn.source),
            summary_input_hash=(
                summary_digest if summaries.get(fn.source) else None
            ),
        )
        for fn in extracted.functions
    ]
    classes = [
        ClassDef(
            name=cls.name,
            qualified_name=cls.qualified_name,
            kind=cls.kind,
            start_line=cls.start_line,
            end_line=cls.end_line,
            start_byte=cls.start_byte,
            end_byte=cls.end_byte,
            source=cls.source,
            source_hash=cls.source_hash,
            parent_class_qname=cls.parent_class_qname,
        )
        for cls in extracted.classes
    ]
    relationships = [
        RelationshipRecord(
            source_type=_source_type(rels.source_qname, extracted),
            source_qname=rels.source_qname,
            relation=rels.relation,
            line_number=rels.line_number,
            resolution=rels.resolution,
            snippet=rels.snippet,
            target_name=rels.target_name,
        )
        for rels in extracted.relationships
    ]
    entity_index = _entity_index_rows(
        extracted, vectors, old_index, model_fingerprint,
    )

    stat = path.stat()
    publish_file_refresh(
        conn, job, file_row, classes, functions, relationships, now_ms,
        indexed_hash=digest, entity_index=entity_index,
        queue_duration_ms=queue_duration,
        process_duration_ms=max(0, now_ms - started),
    )
    conn.execute(
        "UPDATE files SET observed_mtime_ns = ?, observed_size_bytes = ?,"
        " observed_hash = ?, updated_at_ms = ? WHERE folder = ? AND file = ?",
        (stat.st_mtime_ns, stat.st_size, digest, now_ms, folder, file),
    )
    return "completed"


# --- helpers ----------------------------------------------------------------


def _record(conn, job, outcome, error_code, queue_duration, started,
            now_ms, detail=None):
    try:
        record_metric(
            conn,
            metric_kind="job",
            operation=job["command"],
            outcome=outcome,
            component="orchestrator",
            now_ms=now_ms,
            job_id=job["id"],
            job_attempt=job["attempt_count"],
            queue_duration_ms=queue_duration,
            process_duration_ms=max(0, now_ms - started),
            command=job["command"],
            error_code=error_code,
            detail=detail,
        )
    except Exception:
        pass  # observability never alters the job result (plan §5.4)


def _mark_file_error(conn, folder, file, code, now_ms, language=None):
    """Mark a file error/stale without touching its last-good entities."""
    row = conn.execute(
        "SELECT id FROM files WHERE folder = ? AND file = ?", (folder, file)
    ).fetchone()
    if row is None:
        if language is None:
            return
        ensure_file_row(conn, folder, file, language=language,
                        is_test=_is_test(f"{folder}/{file}" if folder
                                         else file), now_ms=now_ms)
    conn.execute(
        "UPDATE files SET status = 'error', last_error = ?,"
        " updated_at_ms = ? WHERE folder = ? AND file = ?",
        (code, now_ms, folder, file),
    )


def _existing_snapshot(conn, file_row):
    """Return ({(type, qname): source_hash}, {(type, qname): index row})."""
    file_id = file_row["id"]
    hashes = {}
    for row in conn.execute(
        "SELECT qualified_name, source_hash FROM functions WHERE file_id = ?",
        (file_id,),
    ):
        hashes[("function", row["qualified_name"])] = row["source_hash"]
    for row in conn.execute(
        "SELECT qualified_name, source_hash FROM classes WHERE file_id = ?",
        (file_id,),
    ):
        hashes[("class", row["qualified_name"])] = row["source_hash"]
    index = {}
    qname_by_id = {}
    for row in conn.execute(
        "SELECT id, qualified_name FROM functions WHERE file_id = ?",
        (file_id,),
    ):
        qname_by_id[("function", row["id"])] = row["qualified_name"]
    for row in conn.execute(
        "SELECT id, qualified_name FROM classes WHERE file_id = ?",
        (file_id,),
    ):
        qname_by_id[("class", row["id"])] = row["qualified_name"]
    for entity_type, entity_id in qname_by_id:
        row = conn.execute(
            "SELECT * FROM entity_index WHERE entity_type = ?"
            " AND entity_id = ?",
            (entity_type, entity_id),
        ).fetchone()
        if row is not None:
            index[(entity_type, qname_by_id[(entity_type, entity_id)])] = row
    return hashes, index


def _entity_index_rows(extracted, vectors, old_index, model_fingerprint):
    """One search row per function/class: fresh vectors for changed
    sources, the retained embedding reused for unchanged sources."""
    rows = []
    for fn in extracted.functions:
        rows.append(_index_row(
            "function", fn.qualified_name, fn.source, fn.source_hash,
            vectors, old_index, model_fingerprint,
        ))
    for cls in extracted.classes:
        rows.append(_index_row(
            "class", cls.qualified_name, cls.source, cls.source_hash,
            vectors, old_index, model_fingerprint,
        ))
    return [row for row in rows if row is not None]


def _index_row(entity_type, qname, source, source_hash, vectors, old_index,
               model_fingerprint):
    vector = vectors.get(source)
    if vector is not None:
        embedding = pack(vector)
        dimensions = len(vector)
    else:
        old = old_index.get((entity_type, qname))
        if old is not None:
            embedding = old["embedding"]
            dimensions = old["embedding_dimensions"]
        else:
            embedding = None  # no inference available; omit the row
    if embedding is None:
        return None
    return {
        "entity_type": entity_type,
        "entity_key": qname,
        "search_text": source,
        "embedding": embedding,
        "dimensions": dimensions,
        "input_hash": hashlib.sha256(
            (model_fingerprint + "\x00" + source_hash).encode("utf-8")
        ).hexdigest(),
        "model_fingerprint": model_fingerprint,
        "embedding_truncated": False,
    }


def _source_type(source_qname, extracted):
    for fn in extracted.functions:
        if fn.qualified_name == source_qname:
            return EntityType.FUNCTION
    for cls in extracted.classes:
        if cls.qualified_name == source_qname:
            return EntityType.CLASS
    return EntityType.FILE


def _is_test(rel_path):
    from indexedcode.scan import _is_test_file

    return _is_test_file(rel_path)
