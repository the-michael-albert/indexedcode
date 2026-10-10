"""SQLite connection, schema DDL, and the OS-level writer lock.

Implements IMPLEMENTATION_PLAN.md §5.1/§5.2: WAL journal, foreign keys on
every connection, a busy timeout, and one writer per database enforced by an
exclusive lock file.
"""

from __future__ import annotations

import fcntl
import sqlite3
from pathlib import Path

BUSY_TIMEOUT_MS = 5000

DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    id                    INTEGER PRIMARY KEY,
    command               TEXT NOT NULL CHECK (command IN
                            ('refresh_file', 'remove_file', 'refresh_module')),
    folder                TEXT NOT NULL,
    file                  TEXT,
    function_name         TEXT,
    line_number           INTEGER CHECK (line_number IS NULL OR line_number >= 1),
    payload_json          TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(payload_json)),
    status                TEXT NOT NULL CHECK (status IN
                            ('pending', 'running', 'retry', 'failed', 'done')),
    desired_revision      INTEGER NOT NULL CHECK (desired_revision > 0),
    lease_token           TEXT,
    lease_expires_at_ms   INTEGER,
    attempt_count         INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    available_at_ms       INTEGER NOT NULL,
    last_error            TEXT,
    created_at_ms         INTEGER NOT NULL,
    updated_at_ms         INTEGER NOT NULL,
    CHECK (status != 'running' OR
      (lease_token IS NOT NULL AND lease_expires_at_ms IS NOT NULL)
    )
);
CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs(status, available_at_ms);
CREATE INDEX IF NOT EXISTS jobs_path_idx ON jobs(folder, file, status);

CREATE TABLE IF NOT EXISTS modules (
    id                    INTEGER PRIMARY KEY,
    parent_module_id      INTEGER REFERENCES modules(id) ON DELETE CASCADE,
    name                  TEXT NOT NULL,
    relative_path         TEXT NOT NULL UNIQUE,
    source_hash           TEXT,
    summary               TEXT,
    updated_at_ms         INTEGER NOT NULL,
    UNIQUE (parent_module_id, name)
);
CREATE INDEX IF NOT EXISTS modules_parent_idx ON modules(parent_module_id);

CREATE TABLE IF NOT EXISTS files (
    id                    INTEGER PRIMARY KEY,
    module_id             INTEGER NOT NULL REFERENCES modules(id) ON DELETE RESTRICT,
    folder                TEXT NOT NULL,
    file                  TEXT NOT NULL,
    language              TEXT,
    is_test               INTEGER NOT NULL DEFAULT 0 CHECK (is_test IN (0, 1)),
    observed_mtime_ns     INTEGER,
    observed_size_bytes   INTEGER CHECK (observed_size_bytes >= 0),
    observed_hash         TEXT,
    indexed_hash          TEXT,
    observed_revision     INTEGER NOT NULL DEFAULT 1 CHECK (observed_revision > 0),
    indexed_revision      INTEGER,
    status                TEXT NOT NULL CHECK (status IN
                            ('pending', 'ready', 'error', 'skipped', 'deleted')),
    skip_reason           TEXT,
    last_error            TEXT,
    updated_at_ms         INTEGER NOT NULL,
    CHECK (indexed_revision IS NULL OR indexed_revision <= observed_revision),
    UNIQUE (folder, file)
);
CREATE INDEX IF NOT EXISTS files_module_idx ON files(module_id);
CREATE INDEX IF NOT EXISTS files_status_idx ON files(status);

CREATE TABLE IF NOT EXISTS classes (
    id                    INTEGER PRIMARY KEY,
    file_id               INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    parent_class_id       INTEGER REFERENCES classes(id) ON DELETE CASCADE,
    name                  TEXT NOT NULL,
    qualified_name        TEXT NOT NULL,
    kind                  TEXT NOT NULL CHECK (kind IN ('class', 'interface', 'struct', 'enum')),
    start_line            INTEGER NOT NULL CHECK (start_line >= 1),
    end_line              INTEGER NOT NULL CHECK (end_line >= start_line),
    start_byte            INTEGER NOT NULL CHECK (start_byte >= 0),
    end_byte              INTEGER NOT NULL CHECK (end_byte >= start_byte),
    source                TEXT NOT NULL,
    source_hash           TEXT NOT NULL,
    summary               TEXT,
    summary_input_hash    TEXT,
    updated_at_ms         INTEGER NOT NULL,
    UNIQUE (file_id, qualified_name, start_byte)
);
CREATE INDEX IF NOT EXISTS classes_file_idx ON classes(file_id);
CREATE INDEX IF NOT EXISTS classes_name_idx ON classes(name);
CREATE INDEX IF NOT EXISTS classes_qualified_idx ON classes(qualified_name);

CREATE TABLE IF NOT EXISTS functions (
    id                    INTEGER PRIMARY KEY,
    file_id               INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    class_id              INTEGER REFERENCES classes(id) ON DELETE CASCADE,
    parent_function_id    INTEGER REFERENCES functions(id) ON DELETE CASCADE,
    name                  TEXT NOT NULL,
    qualified_name        TEXT NOT NULL,
    signature             TEXT NOT NULL,
    start_line            INTEGER NOT NULL CHECK (start_line >= 1),
    end_line              INTEGER NOT NULL CHECK (end_line >= start_line),
    start_byte            INTEGER NOT NULL CHECK (start_byte >= 0),
    end_byte              INTEGER NOT NULL CHECK (end_byte >= start_byte),
    source                TEXT NOT NULL,
    source_hash           TEXT NOT NULL,
    summary               TEXT,
    summary_input_hash    TEXT,
    updated_at_ms         INTEGER NOT NULL,
    UNIQUE (file_id, qualified_name, signature, start_byte)
);
CREATE INDEX IF NOT EXISTS functions_file_idx ON functions(file_id);
CREATE INDEX IF NOT EXISTS functions_class_idx ON functions(class_id);
CREATE INDEX IF NOT EXISTS functions_name_idx ON functions(name);
CREATE INDEX IF NOT EXISTS functions_qualified_idx ON functions(qualified_name);

CREATE TABLE IF NOT EXISTS entity_index (
    id                    INTEGER PRIMARY KEY,
    entity_type           TEXT NOT NULL CHECK (entity_type IN
                            ('module', 'file', 'class', 'function')),
    entity_id             INTEGER NOT NULL,
    search_text           TEXT NOT NULL,
    embedding             BLOB NOT NULL,
    embedding_dimensions  INTEGER NOT NULL CHECK (embedding_dimensions > 0),
    input_hash            TEXT NOT NULL,
    model_fingerprint     TEXT NOT NULL,
    embedding_truncated   INTEGER NOT NULL DEFAULT 0 CHECK (embedding_truncated IN (0, 1)),
    updated_at_ms         INTEGER NOT NULL,
    CHECK (length(embedding) = embedding_dimensions * 4),
    UNIQUE (entity_type, entity_id)
);
CREATE INDEX IF NOT EXISTS entity_index_type_entity_idx
    ON entity_index(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS entity_index_model_idx
    ON entity_index(model_fingerprint, embedding_dimensions);

CREATE TABLE IF NOT EXISTS relationships (
    id                    INTEGER PRIMARY KEY,
    source_type           TEXT NOT NULL CHECK (source_type IN
                            ('module', 'file', 'class', 'function')),
    source_id             INTEGER NOT NULL,
    relation              TEXT NOT NULL CHECK (relation IN
                            ('imports', 'calls', 'inherits', 'implements', 'references')),
    target_type           TEXT CHECK (target_type IS NULL OR target_type IN
                            ('module', 'file', 'class', 'function')),
    target_id             INTEGER,
    target_name           TEXT,
    origin_file_id        INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    line_number           INTEGER NOT NULL CHECK (line_number >= 1),
    resolution            TEXT NOT NULL CHECK (resolution IN
                            ('unresolved', 'syntactic_candidate', 'resolved')),
    snippet               TEXT NOT NULL,
    CHECK (
      (target_id IS NULL) = (target_type IS NULL)
      AND (target_id IS NOT NULL OR target_name IS NOT NULL)
    )
);
CREATE INDEX IF NOT EXISTS relationships_source_idx
    ON relationships(source_type, source_id, relation);
CREATE INDEX IF NOT EXISTS relationships_target_idx
    ON relationships(target_type, target_id, relation);
CREATE INDEX IF NOT EXISTS relationships_name_idx ON relationships(target_name, relation);
CREATE INDEX IF NOT EXISTS relationships_origin_idx ON relationships(origin_file_id);

CREATE TABLE IF NOT EXISTS metrics (
    id                    INTEGER PRIMARY KEY,
    recorded_at_ms        INTEGER NOT NULL,
    metric_kind           TEXT NOT NULL CHECK (metric_kind IN ('request', 'job')),
    operation             TEXT NOT NULL,
    outcome               TEXT NOT NULL CHECK (outcome IN
                            ('success', 'client_error', 'dependency_error',
                             'retry', 'failed', 'cancelled')),
    component             TEXT NOT NULL CHECK (component IN
                            ('orchestrator', 'embedding_worker', 'summary_worker',
                             'remote_embedding', 'remote_summary')),
    request_id            TEXT,
    status_code           INTEGER,
    request_duration_ms   INTEGER CHECK (request_duration_ms IS NULL
                             OR request_duration_ms >= 0),
    request_bytes         INTEGER CHECK (request_bytes IS NULL OR request_bytes >= 0),
    response_bytes        INTEGER CHECK (response_bytes IS NULL OR response_bytes >= 0),
    job_id                INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
    job_attempt           INTEGER CHECK (job_attempt IS NULL OR job_attempt >= 1),
    queue_duration_ms     INTEGER CHECK (queue_duration_ms IS NULL
                             OR queue_duration_ms >= 0),
    process_duration_ms   INTEGER CHECK (process_duration_ms IS NULL
                             OR process_duration_ms >= 0),
    command               TEXT,
    backend               TEXT CHECK (backend IS NULL OR backend IN
                             ('mlx', 'rocm', 'remote')),
    error_code            TEXT,
    detail_json           TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(detail_json)),
    CHECK (
      (metric_kind = 'request' AND request_duration_ms IS NOT NULL) OR
      (metric_kind = 'job' AND job_id IS NOT NULL AND job_attempt IS NOT NULL
       AND queue_duration_ms IS NOT NULL AND process_duration_ms IS NOT NULL)
    )
);
CREATE INDEX IF NOT EXISTS metrics_recorded_idx ON metrics(recorded_at_ms);
CREATE INDEX IF NOT EXISTS metrics_kind_operation_idx
    ON metrics(metric_kind, operation, recorded_at_ms);
CREATE INDEX IF NOT EXISTS metrics_job_idx ON metrics(job_id, job_attempt);
CREATE INDEX IF NOT EXISTS metrics_component_idx ON metrics(component, recorded_at_ms);
"""


class WriterLockError(RuntimeError):
    """Another writer process already holds the database lock."""


def connect(path, writer: bool = False) -> sqlite3.Connection:
    """Open a connection with the plan's mandatory pragmas applied."""
    conn = sqlite3.connect(
        str(path),
        timeout=BUSY_TIMEOUT_MS / 1000,
        isolation_level=None,  # explicit transaction control
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    if writer:
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def initialize(conn: sqlite3.Connection) -> None:
    """Apply the eight-table DDL. Idempotent."""
    conn.executescript(DDL)


class WriterLock:
    """Exclusive OS-level lock (flock) guarding single-writer access."""

    def __init__(self, path):
        self.path = Path(str(path))
        self._fh = None

    def __enter__(self):
        self._fh = open(self.path, "a+")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._fh.close()
            self._fh = None
            raise WriterLockError(
                f"another IndexedCode writer holds {self.path}"
            ) from exc
        return self

    def __exit__(self, *exc_info):
        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None
        return False
