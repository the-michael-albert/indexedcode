"""Opt-in MLX hardware smoke tests (plan Phase 3 gate).

Run with INDEXEDCODE_SMOKE_MLX=1 on an Apple Silicon machine with
mlx-lm and mlx-embeddings installed. Missing hardware or opt-out is
reported as skipped — never silently passed.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("INDEXEDCODE_SMOKE_MLX") != "1",
    reason="opt-in: set INDEXEDCODE_SMOKE_MLX=1 on Apple Silicon to run",
)


def test_mlx_summary_smoke():
    from indexedcode.workers.hardwares import mlx_summary

    summary = mlx_summary(
        prompt="# def add(a, b): return a + b\nsummarize in one line",
        max_output_tokens=64,
    )
    assert isinstance(summary, str) and summary.strip()


def test_mlx_embedding_smoke():
    from indexedcode.embeddings import validate
    from indexedcode.workers.hardwares import mlx_embed

    vectors = mlx_embed(["def add(a, b):", "class Foo:"])
    assert len(vectors) == 2
    for vector in vectors:
        assert validate(vector)  # dimension/finite/nonzero/unit-norm


def test_mlx_end_to_end_summarize_then_embed(tmp_path, real_semantic):
    """Real MLX workers: index once summarizes and embeds every entity."""
    from indexedcode.cli import main
    from indexedcode.db import connect
    from indexedcode.retrieval import _inference_progress, search

    root = tmp_path / "repo"
    root.mkdir()
    (root / "util.py").write_text(
        "def add(a, b):\n    return a + b\n\n\n"
        "class Counter:\n    def bump(self):\n        self.n += 1\n")
    db = tmp_path / "index.sqlite"
    assert main(["index", "once", "--root", str(root), "--db", str(db)]) == 0

    conn = connect(db, writer=False)
    try:
        progress = _inference_progress(conn, None)
        assert progress["stages"]["summary"]["total"] >= 3
        assert progress["complete"], progress
        found = search(conn, str(root), "add two numbers", mode="semantic",
                       embedder=__import__(
                           "indexedcode.workers.hardwares",
                           fromlist=["mlx_embed"]).mlx_embed)
        assert found["results"]
    finally:
        conn.close()
