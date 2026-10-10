"""Shared worker exception hierarchy (plan §4.4)."""

from __future__ import annotations


class WorkerError(Exception):
    """Base class for worker supervision/client failures."""


class WorkerStartupError(WorkerError):
    """A managed worker failed to start or reach authenticated readiness."""


class FingerprintMismatchError(WorkerError):
    """A worker advertised a model fingerprint other than the expected one."""


class WorkerFailedError(WorkerError):
    """A worker exhausted its restart budget and is terminally failed."""


class WorkerAuthError(WorkerError):
    """The worker rejected our credentials (fatal, not retryable)."""


class WorkerTransientError(WorkerError):
    """An overload/transport failure that is safe to retry with backoff."""


class WorkerProtocolError(WorkerError):
    """A non-transient worker protocol violation (e.g. bad payload)."""


class BackendUnavailableError(WorkerError):
    """The requested accelerator backend is not available on this host."""


class MemoryAdmissionError(ValueError):
    """Invalid memory admission configuration (soft must be below hard)."""
