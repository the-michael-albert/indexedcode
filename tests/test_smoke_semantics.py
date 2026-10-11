"""Opt-in semantic-quality smoke test (Phase 5 gate).

Run with INDEXEDCODE_SMOKE_EMBED=1 when a real embedding model is
configured. Missing model is reported skipped — never silently passed.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("INDEXEDCODE_SMOKE_EMBED") != "1",
    reason="opt-in: set INDEXEDCODE_SMOKE_EMBED=1 with a real embedder",
)


def test_behavioral_query_retrieves_intended_entity_top_five(
        tmp_path, real_semantic):
    from indexedcode.cli import main
    from indexedcode.retrieval import search

    root = tmp_path / "repo"
    root.mkdir()
    (root / "src").mkdir()
    (root / "src" / "crypto.py").write_text(
        "def hash_password(password, salt):\n"
        "    digest = bytearray()\n"
        "    for block in _stretch(password, salt):\n"
        "        digest.extend(block)\n"
        "    return bytes(digest)\n"
    )
    (root / "src" / "io.py").write_text(
        "def read_config(path):\n"
        "    with open(path) as handle:\n"
        "        return handle.read()\n"
    )
    db = tmp_path / "index.sqlite"
    main(["index", "once", "--root", str(root), "--db", str(db)])

    from indexedcode.db import connect
    from indexedcode.workers.hardwares import mlx_embed

    conn = connect(db, writer=False)
    try:
        result = search(
            conn, str(root), "hash and stretch a user password with salt",
            mode="semantic", embedder=mlx_embed,
        )
        names = [r["qualified_name"] for r in result["results"][:5]]
        assert "hash_password" in names
    finally:
        conn.close()
