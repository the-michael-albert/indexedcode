"""Reproducible benchmarks (plan Phase 6 gate).

Runs the real pipeline end-to-end on a generated corpus and measures:
initial indexing throughput, warm symbol/vector lookup latency, update
inference counts (unchanged content must re-infer nothing), and idle
watcher cost. All measurements are reported, not promised.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path

from indexedcode.db import WriterLock, connect, initialize
from indexedcode.pipeline import run_until_idle
from indexedcode.reconcile import reconcile
from indexedcode.retrieval import search
from indexedcode.watcher import EventWatcher


class _CountingEmbedder:
    """Deterministic embedder that counts inference calls."""

    def __init__(self):
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        return [_stable_vector(text) for text in texts]


def _stable_vector(text, dimensions=64):
    import hashlib
    import math

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = [(b / 255.0) - 0.5 for b in (digest * 8)[:dimensions]]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def run_benchmarks(root, db_path, *, lookup_samples=50,
                   lookup_function_target=10_000) -> dict:
    """Index ``root`` into ``db_path`` and return the measurement dict."""
    root = Path(root)
    db_path = Path(db_path)
    conn = connect(db_path, writer=True)
    initialize(conn)

    with WriterLock(f"{db_path}.writerlock"):
        # --- initial index ------------------------------------------------
        embedder = _CountingEmbedder()
        started = time.perf_counter()
        reconcile(conn, str(root), now_ms=_now_ms())
        run_until_idle(conn, str(root), embed=embedder,
                       summarize=lambda texts: ["summary"] * len(texts),
                       now_ms=_now_ms())
        initial_seconds = time.perf_counter() - started
        files_indexed = conn.execute(
            "SELECT COUNT(*) AS c FROM files WHERE status = 'ready'"
        ).fetchone()["c"]
        functions_indexed = conn.execute(
            "SELECT COUNT(*) AS c FROM functions"
        ).fetchone()["c"]

        # --- warm lookup latency -------------------------------------------
        symbols = [row["qualified_name"] for row in conn.execute(
            "SELECT qualified_name FROM functions"
            f" LIMIT {lookup_samples}"
        )]
        symbol_timings = []
        for symbol in symbols:
            t0 = time.perf_counter()
            search(conn, str(root), symbol, mode="symbol", limit=5)
            symbol_timings.append((time.perf_counter() - t0) * 1000)
        vector_timings = []
        for symbol in symbols[:max(1, lookup_samples // 5)]:
            row = conn.execute(
                "SELECT search_text FROM entity_index ei"
                " JOIN functions f ON ei.entity_type='function'"
                " AND ei.entity_id = f.id WHERE f.qualified_name = ?",
                (symbol,),
            ).fetchone()
            if row is None:
                continue
            query = row["search_text"][:200]
            t0 = time.perf_counter()
            search(conn, str(root), query, mode="semantic",
                   embedder=lambda texts: [_stable_vector(t)
                                           for t in texts], limit=5)
            vector_timings.append((time.perf_counter() - t0) * 1000)

        # --- update inference counts ----------------------------------------
        first_file = conn.execute(
            "SELECT folder, file FROM files WHERE status = 'ready'"
            " AND language = 'python' ORDER BY id LIMIT 1"
        ).fetchone()
        path = root / (f"{first_file['folder']}/{first_file['file']}"
                       if first_file["folder"] else first_file["file"])
        baseline_calls = len(embedder.calls)
        reconcile(conn, str(root), now_ms=_now_ms())
        run_until_idle(conn, str(root), embed=embedder,
                       summarize=lambda texts: ["summary"] * len(texts),
                       now_ms=_now_ms())
        unchanged_calls = len(embedder.calls) - baseline_calls

        original = path.read_text()
        path.write_text(original + "\n\ndef bench_added_function():\n"
                                "    return 'benchmark'\n")
        changed_calls_before = len(embedder.calls)
        reconcile(conn, str(root), now_ms=_now_ms())
        run_until_idle(conn, str(root), embed=embedder,
                       summarize=lambda texts: ["summary"] * len(texts),
                       now_ms=_now_ms())
        changed_calls = len(embedder.calls) - changed_calls_before
        path.write_text(original)

        # --- idle watcher cost -----------------------------------------------
        watcher = EventWatcher(on_flush=lambda paths: None, debounce_s=0.25)
        t0 = time.perf_counter()
        idle_loops = 0
        while time.perf_counter() - t0 < 0.2:
            watcher.record("src/noop.py", now=time.monotonic())
            watcher.flush(time.monotonic())
            run_until_idle(conn, str(root), embed=embedder,
                           summarize=lambda t: ["summary"] * len(t),
                           now_ms=_now_ms())
            idle_loops += 1
        idle_seconds = time.perf_counter() - t0
        reconcile(conn, str(root), now_ms=_now_ms())
        run_until_idle(conn, str(root), embed=embedder,
                       summarize=lambda texts: ["summary"] * len(texts),
                       now_ms=_now_ms())
        idle_pending = conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE status IN"
            " ('pending', 'retry')"
        ).fetchone()["c"]

    conn.close()

    return {
        "initial": {
            "files_indexed": files_indexed,
            "functions_indexed": functions_indexed,
            "seconds": round(initial_seconds, 4),
            "files_per_second": round(
                files_indexed / initial_seconds, 2
            ) if initial_seconds else 0.0,
        },
        "warm_lookup": {
            "symbol_ms_p50": round(statistics.median(symbol_timings), 4),
            "vector_ms_p50": round(
                statistics.median(vector_timings) if vector_timings else 0.0,
                4,
            ),
            "functions": functions_indexed,
            "lookup_function_target": lookup_function_target,
            "samples": len(symbol_timings),
        },
        "update": {
            "inference_calls_on_unchanged": unchanged_calls,
            "inference_calls_on_changed": changed_calls,
        },
        "idle": {
            "watcher_poll_ms": round(idle_seconds * 1000 / max(1, idle_loops), 4),
            "idle_noop_jobs": idle_pending,
        },
    }


def _now_ms() -> int:
    return int(time.time() * 1000)
