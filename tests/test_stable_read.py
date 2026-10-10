"""Stable file snapshots (plan §7.2.2): stat/read/stat with retries."""

import pytest

from indexedcode.stable import UnstableReadError, read_stable


class FakeStat:
    def __init__(self, size, mtime_ns):
        self.st_size = size
        self.st_mtime_ns = mtime_ns


def test_returns_content_and_sha256(tmp_path):
    path = tmp_path / "a.py"
    path.write_bytes(b"def foo(): pass\n")

    data, digest = read_stable(path)

    assert data == b"def foo(): pass\n"
    assert len(digest) == 64
    import hashlib

    assert digest == hashlib.sha256(b"def foo(): pass\n").hexdigest()


def test_unstable_file_raises_after_attempts(tmp_path):
    path = tmp_path / "a.py"
    path.write_bytes(b"x" * 10)
    sizes = iter([FakeStat(10, 1), FakeStat(11, 1)] * 3)

    with pytest.raises(UnstableReadError):
        read_stable(path, stat=lambda p: next(sizes), attempts=3)


def test_transient_instability_recovers_on_next_attempt(tmp_path):
    path = tmp_path / "a.py"
    path.write_bytes(b"stable")
    # first pass unstable, second pass stable
    stats = iter([
        FakeStat(6, 1), FakeStat(7, 1),   # attempt 1: changed
        FakeStat(6, 2), FakeStat(6, 2),   # attempt 2: stable
    ])

    data, digest = read_stable(path, stat=lambda p: next(stats), attempts=3)

    assert data == b"stable"
    assert digest


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_stable(tmp_path / "nope.py")
