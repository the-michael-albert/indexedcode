"""OS-level writer-lock and single-writer behavior tests (plan §5.1)."""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from indexedcode.db import WriterLock, WriterLockError


def test_second_writer_rejected_same_process(tmp_path):
    lock_path = tmp_path / "index.sqlite.lock"
    with WriterLock(lock_path):
        with pytest.raises(WriterLockError):
            with WriterLock(lock_path):
                pass


def test_lock_released_after_context_exit(tmp_path):
    lock_path = tmp_path / "index.sqlite.lock"
    with WriterLock(lock_path):
        pass
    with WriterLock(lock_path):
        pass  # reacquire succeeds


def test_second_writer_rejected_across_processes(tmp_path):
    """The lock must be OS-enforced, not just in-process state."""
    lock_path = tmp_path / "index.sqlite.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                f"""
                import time
                from indexedcode.db import WriterLock
                with WriterLock({str(lock_path)!r}):
                    print("held", flush=True)
                    time.sleep(5)
                """
            ),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(WriterLockError):
            with WriterLock(lock_path):
                pass
    finally:
        holder.terminate()
        holder.wait(timeout=10)
