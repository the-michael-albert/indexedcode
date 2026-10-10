"""Watcher debouncing (plan §7.1): events are dirty hints; bursts
coalesce; parent representations refresh at most once per interval.

Pure logic with an injected clock — no filesystem event backend needed.
"""

from indexedcode.watcher import EventWatcher, ParentDebouncer


def collected():
    box = []
    return box, lambda paths: box.append(sorted(paths))


def test_event_before_window_is_not_flushed():
    box, on_flush = collected()
    watcher = EventWatcher(on_flush=on_flush, debounce_s=0.25)
    watcher.record("src/a.py", now=0.0)
    assert not watcher.due(now=0.2)
    watcher.flush(now=0.2)
    assert box == []


def test_burst_coalesces_into_single_flush():
    box, on_flush = collected()
    watcher = EventWatcher(on_flush=on_flush, debounce_s=0.25)
    for _ in range(5):
        watcher.record("src/a.py", now=0.0)
    watcher.record("src/b.py", now=0.1)
    assert watcher.due(now=0.26)
    watcher.flush(now=0.26)
    assert box == [["src/a.py", "src/b.py"]]
    watcher.flush(now=0.5)
    assert len(box) == 1  # flushed once, not again


def test_flush_without_events_is_noop():
    box, on_flush = collected()
    watcher = EventWatcher(on_flush=on_flush, debounce_s=0.25)
    watcher.flush(now=1.0)
    assert box == []


def test_events_after_flush_start_a_new_window():
    box, on_flush = collected()
    watcher = EventWatcher(on_flush=on_flush, debounce_s=0.25)
    watcher.record("src/a.py", now=0.0)
    watcher.flush(now=0.3)
    watcher.record("src/a.py", now=0.4)
    assert not watcher.due(now=0.5)
    assert watcher.due(now=0.66)
    watcher.flush(now=0.66)
    assert box == [["src/a.py"], ["src/a.py"]]


def test_due_requires_window_from_first_event():
    box, on_flush = collected()
    watcher = EventWatcher(on_flush=on_flush, debounce_s=0.25)
    watcher.record("src/a.py", now=0.0)
    assert not watcher.due(now=0.24)
    assert watcher.due(now=0.25)


def test_parent_debouncer_allows_one_refresh_per_interval():
    debouncer = ParentDebouncer(interval_s=1.0)
    assert debouncer.should_refresh("src", now=0.0)
    assert not debouncer.should_refresh("src", now=0.5)
    assert not debouncer.should_refresh("src", now=0.99)
    assert debouncer.should_refresh("src", now=1.0)


def test_parent_debouncer_tracks_parents_independently():
    debouncer = ParentDebouncer(interval_s=1.0)
    assert debouncer.should_refresh("src", now=0.0)
    assert not debouncer.should_refresh("src", now=0.2)
    assert debouncer.should_refresh("tests", now=0.2)


def test_parent_debouncer_reset_after_flush():
    debouncer = ParentDebouncer(interval_s=1.0)
    debouncer.should_refresh("src", now=0.0)
    debouncer.mark_refreshed("src", now=0.0)
    assert not debouncer.should_refresh("src", now=0.5)
    assert debouncer.should_refresh("src", now=1.1)
