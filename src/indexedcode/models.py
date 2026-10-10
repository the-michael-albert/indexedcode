"""DAL object shapes and enums mirroring the SQL CHECK constraints.

Every enum member set must match its CHECK constraint in db.py exactly.
Models validate inputs before SQL is reached; Job keeps Row-style subscript
access so existing call sites remain compatible.
"""

from __future__ import annotations

import sqlite3
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

JOB_COMMANDS = ("refresh_file", "remove_file", "refresh_module")
JOB_STATUSES = ("pending", "running", "retry", "failed", "done")
FILE_STATUSES = ("pending", "ready", "error", "skipped", "deleted")
CLASS_KINDS = ("class", "interface", "struct", "enum")
ENTITY_TYPES = ("module", "file", "class", "function")
RELATIONS = ("imports", "calls", "inherits", "implements", "references")
RESOLUTIONS = ("unresolved", "syntactic_candidate", "resolved")
METRIC_KINDS = ("request", "job")
METRIC_OUTCOMES = (
    "success",
    "client_error",
    "dependency_error",
    "retry",
    "failed",
    "cancelled",
)
METRIC_COMPONENTS = (
    "orchestrator",
    "embedding_worker",
    "summary_worker",
    "remote_embedding",
    "remote_summary",
)
BACKENDS_SQL = ("mlx", "rocm", "remote")


class _SqlEnum(str, Enum):
    def __str__(self) -> str:  # keep SQL parameters as plain values
        return self.value


class JobCommand(_SqlEnum):
    REFRESH_FILE = "refresh_file"
    REMOVE_FILE = "remove_file"
    REFRESH_MODULE = "refresh_module"


class JobStatus(_SqlEnum):
    PENDING = "pending"
    RUNNING = "running"
    RETRY = "retry"
    FAILED = "failed"
    DONE = "done"


class FileStatus(_SqlEnum):
    PENDING = "pending"
    READY = "ready"
    ERROR = "error"
    SKIPPED = "skipped"
    DELETED = "deleted"


class ClassKind(_SqlEnum):
    CLASS = "class"
    INTERFACE = "interface"
    STRUCT = "struct"
    ENUM = "enum"


class EntityType(_SqlEnum):
    MODULE = "module"
    FILE = "file"
    CLASS = "class"
    FUNCTION = "function"


class Relation(_SqlEnum):
    IMPORTS = "imports"
    CALLS = "calls"
    INHERITS = "inherits"
    IMPLEMENTS = "implements"
    REFERENCES = "references"


class Resolution(_SqlEnum):
    UNRESOLVED = "unresolved"
    SYNTACTIC_CANDIDATE = "syntactic_candidate"
    RESOLVED = "resolved"


class MetricKind(_SqlEnum):
    REQUEST = "request"
    JOB = "job"


class MetricOutcome(_SqlEnum):
    SUCCESS = "success"
    CLIENT_ERROR = "client_error"
    DEPENDENCY_ERROR = "dependency_error"
    RETRY = "retry"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MetricComponent(_SqlEnum):
    ORCHESTRATOR = "orchestrator"
    EMBEDDING_WORKER = "embedding_worker"
    SUMMARY_WORKER = "summary_worker"
    REMOTE_EMBEDDING = "remote_embedding"
    REMOTE_SUMMARY = "remote_summary"


class Backend(_SqlEnum):
    MLX = "mlx"
    ROCM = "rocm"
    REMOTE = "remote"


class RowModel(BaseModel):
    """Base model usable wherever a sqlite3.Row was passed before."""

    model_config = ConfigDict(frozen=True)

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as exc:
            raise KeyError(key) from exc

    def __conform__(self, protocol):
        # lets Job instances bind directly as an id parameter
        if protocol is sqlite3.PrepareProtocol and hasattr(self, "id"):
            return self.id
        return None


class Job(RowModel):
    """A jobs table row as returned by the DAL."""

    id: int
    command: JobCommand
    folder: str
    file: str | None
    function_name: str | None
    line_number: int | None
    payload_json: str
    status: JobStatus
    desired_revision: int
    lease_token: str | None
    lease_expires_at_ms: int | None
    attempt_count: int
    available_at_ms: int
    last_error: str | None
    created_at_ms: int
    updated_at_ms: int

    def __eq__(self, other):
        # jobs compare by id, including against plain ints, so callers can
        # treat a returned Job exactly like the old row id
        if isinstance(other, Job):
            return self.id == other.id
        if isinstance(other, int):
            return self.id == other
        return NotImplemented

    def __hash__(self):
        return hash(self.id)


class NewJob(RowModel):
    """Validated input for dal.enqueue_job."""

    command: JobCommand
    folder: str
    file: str | None = None
    function_name: str | None = None
    line_number: int | None = Field(default=None, ge=1)
    desired_revision: int = Field(default=1, ge=1)
    payload: dict = Field(default_factory=dict)
    available_at_ms: int = 0
    now_ms: int = 0



class FunctionDef(RowModel):
    """Input shape for one function/method in a file refresh."""

    name: str
    qualified_name: str
    signature: str
    start_line: int = Field(ge=1)
    end_line: int
    start_byte: int = Field(ge=0)
    end_byte: int
    source: str
    source_hash: str
    class_id: int | None = None
    parent_function_id: int | None = None
    # qname linkage resolved inside the publication transaction when the
    # numeric ids are absent (fresh ids are assigned during publication)
    parent_class_qname: str | None = None
    parent_function_qname: str | None = None
    summary: str | None = None
    summary_input_hash: str | None = None

    @model_validator(mode="after")
    def _check_order(self):
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        if self.end_byte < self.start_byte:
            raise ValueError("end_byte must be >= start_byte")
        return self


class ClassDef(RowModel):
    """Input shape for one class-like declaration in a file refresh."""

    name: str
    qualified_name: str
    kind: ClassKind
    start_line: int = Field(ge=1)
    end_line: int
    start_byte: int = Field(ge=0)
    end_byte: int
    source: str
    source_hash: str
    parent_class_id: int | None = None
    parent_class_qname: str | None = None
    summary: str | None = None
    summary_input_hash: str | None = None

    @model_validator(mode="after")
    def _check_order(self):
        if self.end_line < self.start_line:
            raise ValueError("end_line must be >= start_line")
        if self.end_byte < self.start_byte:
            raise ValueError("end_byte must be >= start_byte")
        return self


class RelationshipRecord(RowModel):
    """Input shape for one source-grounded relationship edge."""

    source_type: EntityType
    relation: Relation
    line_number: int = Field(ge=1)
    resolution: Resolution
    snippet: str
    source_id: int | None = None
    source_qname: str | None = None
    target_type: EntityType | None = None
    target_id: int | None = None
    target_name: str | None = None

    @model_validator(mode="after")
    def _check_target(self):
        if (self.source_id is None) == (self.source_qname is None):
            raise ValueError(
                "exactly one of source_id or source_qname must be set"
            )
        if (self.target_id is None) != (self.target_type is None):
            raise ValueError("target_id and target_type must be set together")
        if self.target_id is None and self.target_name is None:
            raise ValueError("a relationship needs a target_id or target_name")
        return self


class EntityIndexInput(RowModel):
    """Input shape for one searchable embedding row."""

    entity_type: EntityType
    entity_id: int
    search_text: str
    embedding: bytes
    dimensions: int = Field(ge=1)
    input_hash: str
    model_fingerprint: str
    embedding_truncated: bool = False
    now_ms: int = 0
