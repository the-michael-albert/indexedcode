"""Stable file snapshots (plan §7.2.2).

stat / read / stat with retries; the caller receives the exact bytes read
plus their SHA-256 so the snapshot and hash can never disagree.
"""

from __future__ import annotations

import hashlib
import os


class UnstableReadError(RuntimeError):
    """The file kept changing while it was being read."""


def read_stable(path, attempts=3, stat=None, read=None):
    """Return ``(data, sha256_hexdigest)`` for a quiescent file.

    ``stat``/``read`` are injectable for deterministic testing; both take
    the path as their only argument.
    """
    stat = stat or os.stat
    if read is None:
        def read(p):  # noqa: E731 - keeps the default local and simple
            with open(p, "rb") as handle:
                return handle.read()

    for _ in range(max(1, attempts)):
        before = stat(path)
        data = read(path)
        after = stat(path)
        if (before.st_size == after.st_size
                and before.st_mtime_ns == after.st_mtime_ns):
            return data, hashlib.sha256(data).hexdigest()
    raise UnstableReadError(f"{path} kept changing while being read")
