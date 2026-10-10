"""Benchmark harness (plan Phase 6 gate).

The harness itself is what's under test here: it must run the real
pipeline end-to-end on a generated corpus and return every documented
measurement key with sane values.
"""

import pytest

from indexedcode.bench import run_benchmarks
from indexedcode.fixture import build_corpus


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    root = tmp_path_factory.mktemp("bench_corpus")
    build_corpus(root, python_modules=4, ts_modules=1, java_modules=1,
                 go_modules=1, rust_modules=1, cpp_modules=1, js_modules=1)
    return root


def test_benchmark_reports_initial_throughput(corpus, tmp_path):
    result = run_benchmarks(corpus, tmp_path / "bench.sqlite")
    initial = result["initial"]
    assert initial["files_indexed"] >= 10
    assert initial["seconds"] > 0
    assert initial["files_per_second"] > 0
    assert initial["functions_indexed"] > 50


def test_benchmark_reports_warm_lookup_latencies(corpus, tmp_path):
    result = run_benchmarks(corpus, tmp_path / "bench.sqlite")
    warm = result["warm_lookup"]
    assert warm["symbol_ms_p50"] >= 0
    assert warm["vector_ms_p50"] >= 0
    assert warm["functions"] > 0


def test_benchmark_reports_update_inference_counts(corpus, tmp_path):
    result = run_benchmarks(corpus, tmp_path / "bench.sqlite")
    update = result["update"]
    assert update["inference_calls_on_unchanged"] == 0
    assert update["inference_calls_on_changed"] >= 1


def test_benchmark_reports_idle_watcher_cost(corpus, tmp_path):
    result = run_benchmarks(corpus, tmp_path / "bench.sqlite")
    idle = result["idle"]
    assert idle["watcher_poll_ms"] >= 0
    assert idle["idle_noop_jobs"] == 0  # an idle system does no work


def test_benchmark_leaves_no_pending_jobs(corpus, tmp_path):
    from indexedcode.db import connect

    db = tmp_path / "bench2.sqlite"
    run_benchmarks(corpus, db)
    conn = connect(db)
    try:
        pending = conn.execute(
            "SELECT COUNT(*) AS c FROM jobs WHERE status IN"
            " ('pending', 'retry')"
        ).fetchone()["c"]
        assert pending == 0
    finally:
        conn.close()
