"""Watcher debouncing (plan §7.1).

Events are dirty hints, not commands: bursts coalesce, and a flush fires
once per debounce window measured from the first event. Parent
representations (file/module) refresh at most once per parent per
interval, so a burst of child edits produces at most one parent refresh.

Pure logic with an injected clock; the platform event source (watchfiles)
is attached by the Phase 5 orchestrator CLI.
"""

from __future__ import annotations


class EventWatcher:
    """Coalesce file-event bursts into debounced flushes."""

    def __init__(self, on_flush, debounce_s=0.25):
        self._on_flush = on_flush
        self._debounce_s = debounce_s
        self._pending: set[str] = set()
        self._window_start: float | None = None

    def record(self, path, now):
        """Record one dirty hint at time ``now`` (seconds)."""
        if self._window_start is None:
            self._window_start = now
        self._pending.add(path)

    def due(self, now):
        """True if the window has elapsed and a flush is pending."""
        return (
            bool(self._pending)
            and self._window_start is not None
            and now - self._window_start >= self._debounce_s
        )

    def flush(self, now):
        """Deliver the pending set if due; starts a fresh window after."""
        if not self.due(now):
            return
        self._on_flush(self._pending)
        self._pending = set()
        self._window_start = None


class ParentDebouncer:
    """Allow at most one parent refresh per key per interval.

    A positive answer latches the decision time, so repeated asks within
    the interval are refused without an explicit mark; ``mark_refreshed``
    records the actual refresh for callers that schedule asynchronously.
    """

    def __init__(self, interval_s=1.0):
        self._interval_s = interval_s
        self._last_refreshed: dict[str, float] = {}

    def should_refresh(self, key, now):
        last = self._last_refreshed.get(key)
        if last is None or now - last >= self._interval_s:
            self._last_refreshed[key] = now
            return True
        return False

    def mark_refreshed(self, key, now):
        self._last_refreshed[key] = now
