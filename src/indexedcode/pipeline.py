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

import logging
import sqlite3
import time
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

logger = logging.getLogger(__name__)

_PROCESS_COMMANDS = ("refresh_file", "remove_file")


def run_until_idle(conn, root, *, embed=None, summarize=None, now_ms=0,
                   step_ms=1000, max_iterations=10_000,
                   model_fingerprint="unpinned",
                   summary_fingerprint="unpinned"):
    """Process every due job (advancing past retry delays) until idle.

    Returns a summary dict of outcome counts. Shared by the CLI's
    ``index once`` and the orchestrator loop.
    """
    now = now_ms
    counts = {}
    for _ in range(max_iterations):
        outcome = process_next_job(
            conn, root, embed=embed, summarize=summarize, now_ms=now,
            retry_base_ms=1, retry_cap_ms=1, jitter=lambda: 0,
            model_fingerprint=model_fingerprint,
            summary_fingerprint=summary_fingerprint,
        )
        if outcome is None:
            next_due = conn.execute(
                "SELECT MIN(available_at_ms) AS next FROM jobs"
                " WHERE status IN ('pending', 'retry')"
            ).fetchone()["next"]
            if next_due is None:
                return counts
            now = max(now + step_ms, next_due + 1)
        else:
            counts[outcome] = counts.get(outcome, 0) + 1
            now += step_ms
    raise RuntimeError("pipeline never went idle")


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
    summary_fingerprint="unpinned",
):
    """Claim and run the next due job; return the outcome string or None.

    Outcomes: ``"completed"``, ``"removed"``, ``"retried"``, ``"failed"``.
    ``embed`` maps a list of texts to a list of float vectors; ``summarize``
    maps a list of texts to a list of summary strings.

    Inference is a two-stage pipeline: each changed entity is summarized
    first and the *generated summary* is what gets embedded. ``embed``
    therefore requires ``summarize``; a failed summary fails the file's
    refresh (retryable) rather than embedding raw source instead.
    """
    if embed is not None and summarize is None:
        raise ValueError(
            "embedding requires summarization: summaries are the embedding"
            " input, raw source is never embedded")
    job = claim_next_job(conn, now_ms=now_ms, lease_ms=lease_ms)
    if job is None:
        return None

    started = now_ms
    queue_duration = max(0, started - job["created_at_ms"])
    t0 = time.perf_counter()
    label = (f"{job.folder}/{job.file}" if job.folder
             else (job.file or job.command))
    job_label = f"#{job.id} {job.command} {label}"
    logger.info("RUNNING  %s (attempt %d/%d)", job_label,
                job["attempt_count"], MAX_ATTEMPTS)

    def finish_retry(code, message, detail=None):
        if job["attempt_count"] >= MAX_ATTEMPTS:
            logger.error("FAILED   %s: %s: %s (gave up after %d attempts)",
                         job_label, code, message, job["attempt_count"])
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
        logger.warning("RETRY    %s: %s: %s (next attempt in %d ms)",
                       job_label, code, message, delay)
        retry_job(conn, job.id, job.lease_token, f"{code}: {message}",
                  now_ms + delay, now_ms)
        _record(conn, job, "retry", code, queue_duration,
                started, now_ms, detail)
        return "retried"

    if job.command not in _PROCESS_COMMANDS:
        logger.error("FAILED   %s: unsupported command", job_label)
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
        logger.info("DONE     %s", job_label)
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
    old_hashes, old_summaries, old_index = _existing_snapshot(conn, file_row)
    entities = [("function", fn) for fn in extracted.functions]
    entities += [("class", cls) for cls in extracted.classes]
    # occurrence keys tell same-named definitions apart; stored data is
    # matched to them by source hash first so reordering/edits keep caches
    keys = _occurrence_keys(extracted)
    old_of = _match_old(entities, keys, old_hashes)

    # stage 1: summaries. An entity keeps its stored summary while it was
    # produced from the same source by the same summary model.
    summaries: dict[tuple, tuple[str, str]] = {}  # key -> (text, identity)
    to_summarize = []
    for kind, ent in entities:
        key = keys[id(ent)]
        old = old_summaries.get(old_of.get(key))
        identity = _summary_identity(summary_fingerprint, ent.source_hash)
        if summarize is None:
            # no summarizer: keep a summary only while its source is intact
            if (old is not None and old[0] and old[1]
                    and old_hashes.get(old_of.get(key)) == ent.source_hash):
                summaries[key] = old
        elif old is not None and old[0] and old[1] == identity:
            summaries[key] = old
        else:
            to_summarize.append((key, ent, identity))

    # stage 2: vectors, computed from summaries only
    def needs_vector(key):
        if embed is None or key not in summaries:
            return False
        old = old_index.get(old_of.get(key))
        return (old is None or old["model_fingerprint"] != model_fingerprint
                or old["input_hash"]
                != _vector_identity(model_fingerprint, summaries[key][1]))

    vectors: dict[tuple, list[float]] = {}
    inference_calls = 0
    try:
        if to_summarize:
            texts = [ent.source for _, ent, _ in to_summarize]
            generated = list(summarize(texts))
            inference_calls += 1
            if len(generated) != len(texts):
                raise ValueError(
                    f"summarizer returned {len(generated)} summaries for"
                    f" {len(texts)} entities")
            for (key, _ent, identity), text in zip(to_summarize, generated):
                if not isinstance(text, str) or not text.strip():
                    raise ValueError(f"empty summary for {key[1]}")
                summaries[key] = (text.strip(), identity)
        to_embed = [key for key in
                    (keys[id(ent)] for _, ent in entities)
                    if needs_vector(key)]
        if to_embed:
            embedded = embed([summaries[key][0] for key in to_embed])
            inference_calls += 1
            if len(embedded) != len(to_embed):
                raise ValueError(
                    f"embedder returned {len(embedded)} vectors for"
                    f" {len(to_embed)} summaries")
            vectors = dict(zip(to_embed, embedded))
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
            summary=_summary_text(summaries, keys, fn),
            summary_input_hash=_summary_id(summaries, keys, fn),
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
            summary=_summary_text(summaries, keys, cls),
            summary_input_hash=_summary_id(summaries, keys, cls),
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
        extracted, vectors, summaries, old_index, model_fingerprint,
        keys, old_of,
    )

    stat = path.stat()
    try:
        publish_file_refresh(
            conn, job, file_row, classes, functions, relationships, now_ms,
            indexed_hash=digest, entity_index=entity_index,
            queue_duration_ms=queue_duration,
            process_duration_ms=max(0, now_ms - started),
        )
    except (sqlite3.Error, ValueError, KeyError) as exc:
        # publication rolled back: the last-good snapshot is intact. Isolate
        # the failure to this job so the rest of the queue keeps draining;
        # attempts stay bounded by finish_retry.
        logger.error("PUBLISH  %s failed: %s: %s", job_label,
                     type(exc).__name__, exc)
        _mark_file_error(conn, folder, file, "publish_error", now_ms,
                         language=language)
        return finish_retry(
            "publish_error", f"publication failed: {type(exc).__name__}: {exc}",
        )
    conn.execute(
        "UPDATE files SET observed_mtime_ns = ?, observed_size_bytes = ?,"
        " observed_hash = ?, updated_at_ms = ? WHERE folder = ? AND file = ?",
        (stat.st_mtime_ns, stat.st_size, digest, now_ms, folder, file),
    )
    logger.info("DONE     %s (%d functions, %d classes, %d ms)", job_label,
                len(functions), len(classes),
                round((time.perf_counter() - t0) * 1000))
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


def _summary_identity(summary_fingerprint, source_hash):
    """Identity of a summary: the summary model and the source it read."""
    return f"{summary_fingerprint}:{source_hash}"


def _vector_identity(model_fingerprint, summary_identity):
    """Identity of a vector: the embedding model and the summary embedded."""
    return f"{model_fingerprint}:{summary_identity}"


def _summary_text(summaries, keys, ent):
    found = summaries.get(keys[id(ent)])
    return found[0] if found else None


def _summary_id(summaries, keys, ent):
    found = summaries.get(keys[id(ent)])
    return found[1] if found else None


def _occurrence_keys(extracted):
    """Map ``id(entity)`` -> ``(kind, qualified_name, ordinal)``.

    A qualified name is not unique within a file (e.g. the same helper defined
    in two branches of one function); the ordinal is the entity's position
    among same-named definitions in source order.
    """
    keys, seen = {}, {}
    for kind, entities in (("function", extracted.functions),
                           ("class", extracted.classes)):
        for ent in entities:
            base = (kind, ent.qualified_name)
            ordinal = seen.get(base, 0)
            seen[base] = ordinal + 1
            keys[id(ent)] = (*base, ordinal)
    return keys


def _match_old(entities, keys, old_hashes):
    """Pair each new occurrence key with the stored one it descends from.

    Within a (kind, qualified_name) group, identical source hashes pair first
    (so inserting, removing or reordering duplicates keeps correct data);
    leftovers pair by position. Unmatched new keys get no entry.
    """
    old_by_base = {}
    for okey in old_hashes:
        old_by_base.setdefault(okey[:2], []).append(okey)
    new_by_base = {}
    for _, ent in entities:
        nkey = keys[id(ent)]
        new_by_base.setdefault(nkey[:2], []).append((nkey, ent.source_hash))
    matched = {}
    for base, news in new_by_base.items():
        free = sorted(old_by_base.get(base, ()))
        rest = []
        for nkey, digest in news:
            hit = next((o for o in free if old_hashes[o] == digest), None)
            if hit is None:
                rest.append(nkey)
            else:
                free.remove(hit)
                matched[nkey] = hit
        for nkey, okey in zip(rest, free):
            matched[nkey] = okey
    return matched


def _existing_snapshot(conn, file_row):
    """Return the stored state of a file's entities, keyed by (kind, qname, ordinal):
    ({source_hash}, {(summary, summary_input_hash)}, {entity_index row})."""
    file_id = file_row["id"]
    hashes, summaries, index = {}, {}, {}
    for kind, table in (("function", "functions"), ("class", "classes")):
        seen = {}
        for row in conn.execute(
            f"SELECT id, qualified_name, source_hash, summary,"
            f" summary_input_hash FROM {table} WHERE file_id = ?"
            f" ORDER BY id",
            (file_id,),
        ):
            ordinal = seen.get(row["qualified_name"], 0)
            seen[row["qualified_name"]] = ordinal + 1
            key = (kind, row["qualified_name"], ordinal)
            hashes[key] = row["source_hash"]
            summaries[key] = (row["summary"], row["summary_input_hash"])
            vector = conn.execute(
                "SELECT * FROM entity_index WHERE entity_type = ?"
                " AND entity_id = ?",
                (kind, row["id"]),
            ).fetchone()
            if vector is not None:
                index[key] = vector
    return hashes, summaries, index


def _entity_index_rows(extracted, vectors, summaries, old_index,
                       model_fingerprint, keys, old_of):
    """One search row per function/class. Vectors are embeddings of the
    generated summary; ``search_text`` stays the raw source so lexical
    retrieval is unaffected. An unchanged retained vector is reused."""
    rows = []
    for kind, entities in (("function", extracted.functions),
                           ("class", extracted.classes)):
        for ent in entities:
            rows.append(_index_row(
                kind, ent, vectors, summaries, old_index, model_fingerprint,
                keys[id(ent)], old_of))
    return [row for row in rows if row is not None]


def _index_row(kind, ent, vectors, summaries, old_index, model_fingerprint,
               key, old_of):
    vector = vectors.get(key)
    if vector is not None:
        embedding = pack(vector)
        dimensions = len(vector)
        input_hash = _vector_identity(model_fingerprint, summaries[key][1])
    else:
        old = old_index.get(old_of.get(key))
        if old is None:
            return None  # no inference available; omit the row
        embedding = old["embedding"]
        dimensions = old["embedding_dimensions"]
        # a retained vector keeps the identity of what produced it, so a
        # later run notices it is stale and recomputes it
        model_fingerprint = old["model_fingerprint"]
        input_hash = old["input_hash"]
    return {
        "entity_type": kind,
        "entity_key": ent.qualified_name,
        "entity_ordinal": key[2],
        "search_text": ent.source,
        "embedding": embedding,
        "dimensions": dimensions,
        "input_hash": input_hash,
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
