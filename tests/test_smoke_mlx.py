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
