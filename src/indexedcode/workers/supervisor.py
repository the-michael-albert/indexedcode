"""Managed worker supervision (plan §4).

For every managed worker the orchestrator resolves the backend, creates a
private loopback address plus a per-run authentication secret, starts the
worker as a child process, waits for its authenticated readiness and
model-fingerprint response, restarts it after unexpected exits within a
restart budget, and stops it gracefully (SIGTERM, then SIGKILL only after
the configured grace period).
"""

from __future__ import annotations

import os
import secrets
import socket
import subprocess
import time

from indexedcode.workers import adapters
from indexedcode.workers.client import WorkerClient
from indexedcode.workers.errors import (
    FingerprintMismatchError,
    WorkerError,
    WorkerFailedError,
    WorkerStartupError,
    WorkerTransientError,
)

_WORKER_PORT_ENV = "INDEXEDCODE_WORKER_PORT"
_WORKER_SECRET_ENV = "INDEXEDCODE_WORKER_SECRET"


def _free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ManagedWorker:
    """Supervise one local loopback inference worker child process."""

    def __init__(
        self,
        name: str,
        command: list[str],
        expected_fingerprint: str,
        startup_timeout_s: float = 30.0,
        restart_budget: int = 1,
        backoff_s: tuple = (1.0,),
        stop_grace_s: float = 10.0,
        poll_interval_s: float = 0.5,
        backend: str | None = None,
    ):
        self.name = name
        self.command = list(command)
        self.expected_fingerprint = expected_fingerprint
        self.startup_timeout_s = startup_timeout_s
        self.restart_budget = restart_budget
        self.backoff_s = tuple(backoff_s) or (1.0,)
        self.stop_grace_s = stop_grace_s
        self.poll_interval_s = poll_interval_s
        self.backend = backend

        self.state = "created"
        self.pid: int | None = None
        self.secret: str | None = None
        self.base_url: str | None = None
        self.model_info = None
        self.restart_count = 0
        self._proc: subprocess.Popen | None = None
        self._terminal_failure = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self.state in ("ready", "starting", "draining"):
            raise WorkerStartupError(
                f"{self.name}: already started (state={self.state})"
            )
        if self.backend is not None:
            adapters.backend_available(self.backend)
        self.secret = secrets.token_urlsafe(32)
        self._spawn_and_wait_ready()

    def poll(self) -> None:
        """Check the child; restart or fail it after unexpected exits."""
        if self._terminal_failure:
            raise WorkerFailedError(
                f"{self.name}: restart budget"
                f" ({self.restart_budget}) exhausted"
            )
        if self.state not in ("ready", "starting") or self._proc is None:
            return
        if self._proc.poll() is not None:
            self._handle_unexpected_exit()
            if self._terminal_failure:
                raise WorkerFailedError(
                    f"{self.name}: restart budget"
                    f" ({self.restart_budget}) exhausted"
                )

    def begin_drain(self) -> None:
        """Stop accepting new work; a crash during drain is not restarted."""
        if self.state == "ready":
            self.state = "draining"

    def stop(self) -> None:
        if self.state == "stopped":
            return
        self._terminate_child()
        if self.state != "failed":
            self.state = "stopped"

    # -- introspection -----------------------------------------------------

    def status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "restart_count": self.restart_count,
            "restart_budget": self.restart_budget,
            "expected_fingerprint": self.expected_fingerprint,
            "base_url": self.base_url,
            "pid": self.pid,
        }

    # -- internals ---------------------------------------------------------

    def _spawn_and_wait_ready(self) -> None:
        port = _free_loopback_port()
        self.base_url = f"http://127.0.0.1:{port}"
        env = dict(os.environ)
        env[_WORKER_PORT_ENV] = str(port)
        env[_WORKER_SECRET_ENV] = self.secret
        self.state = "starting"
        self._proc = subprocess.Popen(
            self.command,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.pid = self._proc.pid
        try:
            self._wait_ready()
        except Exception:
            self._terminate_child()
            self.state = "failed"
            raise
        self.state = "ready"

    def _wait_ready(self) -> None:
        client = WorkerClient(
            base_url=self.base_url,
            secret=self.secret,
            expected_fingerprint=self.expected_fingerprint,
            timeout_s=min(2.0, max(self.poll_interval_s, 0.1)),
        )
        deadline = time.monotonic() + self.startup_timeout_s
        while True:
            if self._proc.poll() is not None:
                raise WorkerStartupError(
                    f"{self.name}: worker process exited with code"
                    f" {self._proc.returncode} before readiness"
                )
            try:
                self.model_info = client.model_info()
                return
            except WorkerTransientError:
                pass  # not accepting connections yet
            except FingerprintMismatchError:
                raise
            except WorkerError as exc:
                raise WorkerStartupError(
                    f"{self.name}: readiness handshake failed: {exc}"
                ) from exc
            if time.monotonic() >= deadline:
                raise WorkerStartupError(
                    f"{self.name}: worker not ready within"
                    f" {self.startup_timeout_s}s"
                )
            time.sleep(self.poll_interval_s)

    def _handle_unexpected_exit(self) -> None:
        assert self._proc is not None
        self._proc.wait()  # reap
        if self.state in ("draining", "stopped", "failed"):
            return
        if self.restart_count >= self.restart_budget:
            self.state = "failed"
            self._terminal_failure = True
            return
        self.restart_count += 1
        delay = self.backoff_s[
            min(self.restart_count - 1, len(self.backoff_s) - 1)
        ]
        time.sleep(delay)
        try:
            self._spawn_and_wait_ready()
        except WorkerError:
            self.state = "failed"
            self._terminal_failure = True

    def _terminate_child(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=self.stop_grace_s)
                return
            except subprocess.TimeoutExpired:
                proc.kill()
        try:
            proc.wait(timeout=self.stop_grace_s)
        except subprocess.TimeoutExpired:
            pass
