"""Model-spec resolution: local ``.gguf`` vs ``repo_id:glob`` (plan §3).

The suffix cannot tell them apart (``repo:*q8_0.gguf`` also ends in
``.gguf``), so both the loader and the prefetcher share one predicate.
"""

from __future__ import annotations

from indexedcode.workers import hardwares
from indexedcode.workers.serve import prefetch


def test_repo_globs_are_not_local(tmp_path):
    assert not hardwares.is_local_gguf("Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF:*q8_0.gguf")
    assert not hardwares.is_local_gguf("second-state/All-MiniLM-L6-v2-Embedding-GGUF:*Q8_0.gguf")


def test_existing_file_is_local(tmp_path):
    gguf = tmp_path / "my-model.gguf"
    gguf.write_bytes(b"GGUF")
    assert hardwares.is_local_gguf(str(gguf))
    # a colon in the path must not turn an existing file into a repo spec
    odd = tmp_path / "od d:y.gguf"
    odd.write_bytes(b"GGUF")
    assert hardwares.is_local_gguf(str(odd))


def test_bare_name_without_colon_is_local():
    # not an existing file, but shaped like a path: loading (not
    # downloading) must own the "does not exist" error
    assert hardwares.is_local_gguf("/models/missing.gguf")


def test_prefetch_skips_local_specs(tmp_path, monkeypatch):
    import sys
    import types

    fake_hub = types.ModuleType("huggingface_hub")

    def _no_download(*args, **kwargs):
        raise AssertionError("prefetch must not download a local model")

    fake_hub.snapshot_download = _no_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    gguf = tmp_path / "my-model.gguf"
    gguf.write_bytes(b"GGUF")
    prefetch(str(gguf))
    prefetch("/models/missing.gguf")
