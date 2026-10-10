"""Managed worker supervision lifecycle (plan §4).

Covers: startup + authenticated readiness + fingerprint validation,
readiness failure, fingerprint mismatch refusal, unexpected exit with
restart/backoff and budget exhaustion, graceful shutdown, force-kill after
grace, and status reporting. Uses only the deterministic fake worker.
"""

import os
import signal
import subprocess
import sys
import time

import pytest

from indexedcode.workers.fingerprint import model_fingerprint
from indexedcode.workers.protocol import ModelInfo
from indexedcode.workers.supervisor import (
    FingerprintMismatchError,
    ManagedWorker,
    WorkerFailedError,
    WorkerStartupError,
)

from tests.conftest import (
    FAKE_FINGERPRINT,
    FAKE_WORKER,
    free_port,
    pid_alive,
    wait_until,
)

EXPECTED_FP = model_fingerprint(
    model="fake-embed", revision="r1", backend="mlx", dimension=3,
)


def make_worker(mode="ready", **kwargs):
    defaults = dict(
        name="embedding",
        command=[sys.executable, FAKE_WORKER, "--mode", mode],
        expected_fingerprint=EXPECTED_FP,
        startup_timeout_s=10.0,
        restart_budget=1,
        backoff_s=(0.05,),
        stop_grace_s=5.0,
        poll_interval_s=0.05,
    )
    defaults.update(kwargs)
    return ManagedWorker(**defaults)


def test_start_reaches_ready_and_validates_fingerprint():
    worker = make_worker()
    try:
        worker.start()
        assert worker.state == "ready"
        assert isinstance(worker.model_info, ModelInfo)
        assert worker.model_info.fingerprint == EXPECTED_FP
        assert worker.pid is not None and pid_alive(worker.pid)
    finally:
        worker.stop()


def test_readiness_failure_raises_startup_error_and_kills_child():
    worker = make_worker(mode="hang", startup_timeout_s=0.5)
    with pytest.raises(WorkerStartupError):
        worker.start()
    assert worker.state == "failed"
    assert worker.pid is None or not pid_alive(worker.pid)


def test_spawn_immediate_exit_is_a_startup_failure():
    worker = make_worker(mode="exit", startup_timeout_s=2.0)
    with pytest.raises(WorkerStartupError):
        worker.start()
    assert worker.state == "failed"


def test_fingerprint_mismatch_refuses_worker_and_kills_child():
    worker = make_worker(mode="mismatch")
    with pytest.raises(FingerprintMismatchError):
        worker.start()
    assert worker.state == "failed"
    assert worker.pid is None or not pid_alive(worker.pid)


def test_worker_secret_is_generated_not_in_command_and_not_in_status():
    worker = make_worker()
    worker.start()
    try:
        secret = worker.secret
        assert isinstance(secret, str) and len(secret) >= 32
        # never leaks into argv (visible in process listings)
        cmdline = subprocess.run(
            ["ps", "-o", "command=", "-p", str(worker.pid)],
            capture_output=True, text=True,
        ).stdout
        assert secret not in cmdline
        assert secret not in repr(worker.status())
    finally:
        worker.stop()


def test_unexpected_exit_restarts_within_budget():
    # fake worker in "exit" mode dies before readiness only when told to;
    # emulate a crash after readiness by stopping the process ourselves.
    worker = make_worker(mode="ready", restart_budget=1, backoff_s=(0.05,))
    worker.start()
    original_pid = worker.pid
    try:
        os.kill(original_pid, signal.SIGKILL)

        def restarted():
            try:
                worker.poll()
            except WorkerFailedError:
                pass
            return worker.state == "ready" and worker.pid != original_pid

        assert wait_until(restarted, timeout=10), (
            f"expected restart, state={worker.state!r}"
        )
        assert worker.restart_count == 1
        assert pid_alive(worker.pid)
    finally:
        worker.stop()


def test_backoff_exhaustion_marks_worker_failed():
    worker = make_worker(mode="ready", restart_budget=1, backoff_s=(0.05,))
    worker.start()
    first_pid = worker.pid

    def pump():
        try:
            worker.poll()
        except WorkerFailedError:
            pass
        return True

    try:
        # crash 1: within budget -> restarted with a new pid
        os.kill(first_pid, signal.SIGKILL)
        assert wait_until(
            lambda: pump()
            and worker.restart_count == 1
            and worker.state == "ready",
            timeout=10,
        ), f"expected restart, state={worker.state!r}"
        assert worker.pid != first_pid
        # crash 2: budget exhausted -> terminally failed
        os.kill(worker.pid, signal.SIGKILL)
        assert wait_until(
            lambda: pump() and worker.state == "failed", timeout=10
        )
        assert not pid_alive(first_pid)
        with pytest.raises(WorkerFailedError):
            worker.poll()
    finally:
        worker.stop()


def test_graceful_stop_terminates_child_quickly():
    worker = make_worker()
    worker.start()
    pid = worker.pid
    started = time.monotonic()
    worker.stop()
    assert time.monotonic() - started < 5.0
    assert worker.state == "stopped"
    assert not pid_alive(pid)


def test_stop_is_idempotent():
    worker = make_worker()
    worker.start()
    worker.stop()
    worker.stop()
    assert worker.state == "stopped"


def test_stop_force_kills_after_grace_period():
    worker = make_worker(mode="ignore_sigterm", stop_grace_s=0.5)
    worker.start()
    pid = worker.pid
    started = time.monotonic()
    worker.stop()
    elapsed = time.monotonic() - started
    assert not pid_alive(pid)
    assert elapsed >= 0.5
    assert worker.state == "stopped"


def test_drain_prevents_restart_of_crashed_worker():
    worker = make_worker()
    worker.start()
    pid = worker.pid
    worker.begin_drain()  # no new work; shutdown in progress
    os.kill(pid, signal.SIGKILL)
    time.sleep(0.3)
    worker.poll()
    assert worker.state in ("draining", "stopped")
    worker.stop()
    assert not pid_alive(pid)


def test_status_reports_state_restarts_and_budget():
    worker = make_worker(restart_budget=3)
    worker.start()
    try:
        status = worker.status()
        assert status["name"] == "embedding"
        assert status["state"] == "ready"
        assert status["restart_count"] == 0
        assert status["restart_budget"] == 3
        assert "expected_fingerprint" in status
    finally:
        worker.stop()


def test_managed_worker_binds_only_loopback():
    worker = make_worker()
    worker.start()
    try:
        assert worker.base_url.startswith("http://127.0.0.1:")
    finally:
        worker.stop()


def test_start_twice_is_an_error():
    worker = make_worker()
    worker.start()
    try:
        with pytest.raises(WorkerStartupError):
            worker.start()
    finally:
        worker.stop()
