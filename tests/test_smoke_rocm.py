"""Opt-in ROCm hardware smoke tests (plan Phase 3 gate).

Run with INDEXEDCODE_SMOKE_ROCM=1 on a HIP-enabled AMD/Linux host.
Missing hardware or opt-out is reported as skipped — never silently
passed.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("INDEXEDCODE_SMOKE_ROCM") != "1",
    reason="opt-in: set INDEXEDCODE_SMOKE_ROCM=1 on a ROCm host to run",
)


def test_rocm_summary_smoke():
    from indexedcode.workers.hardwares import rocm_summary

    summary = rocm_summary(
        prompt="# def add(a, b): return a + b\nsummarize in one line",
        max_output_tokens=64,
    )
    assert isinstance(summary, str) and summary.strip()


def test_rocm_embedding_smoke():
    from indexedcode.embeddings import validate
    from indexedcode.workers.hardwares import rocm_embed

    vectors = rocm_embed(["def add(a, b):", "class Foo:"])
    assert len(vectors) == 2
    for vector in vectors:
        assert validate(vector)
