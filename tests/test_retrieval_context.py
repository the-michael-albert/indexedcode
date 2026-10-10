"""Exact source context: entity fetch, freshness, explicit disk reads,
truncation/cursors, and traversal/symlink rejection (plan §8.1)."""

import pytest

from indexedcode.retrieval import ContextError, get_context, search


def find_function(conn, root, qualified_name):
    row = conn.execute(
        "SELECT id FROM functions WHERE qualified_name = ?", (qualified_name,)
    ).fetchone()
    assert row is not None
    return row["id"]


def test_entity_context_returns_exact_indexed_source(indexed_repo):
    conn, root = indexed_repo
    entity_id = find_function(conn, root, "parse_config")
    result = get_context(
        conn, str(root), entity_type="function", entity_id=entity_id,
    )
    assert result["origin"] == "index"
    assert result["freshness"] == "current"
    assert result["source"] == (
        "def parse_config(text):\n    return json.loads(text)"
    )
    assert result["path"] == "src/config_loader.py"
    assert result["enclosing"] is None


def test_entity_context_reports_stale_after_disk_edit(indexed_repo):
    conn, root = indexed_repo
    entity_id = find_function(conn, root, "parse_config")
    (root / "src" / "config_loader.py").write_text(
        "def parse_config(text, strict=False):\n    return text\n"
    )
    result = get_context(
        conn, str(root), entity_type="function", entity_id=entity_id,
    )
    assert result["freshness"] == "stale"
    assert result["origin"] == "index"  # last indexed snapshot, marked stale


def test_explicit_path_read_returns_current_disk_source(indexed_repo):
    conn, root = indexed_repo
    (root / "src" / "new_file.py").write_text("def brand_new():\n    pass\n")
    result = get_context(
        conn, str(root), path="src/new_file.py",
    )
    assert result["origin"] == "disk"
    assert "brand_new" in result["source"]


def test_explicit_line_range_is_bounded(indexed_repo):
    conn, root = indexed_repo
    result = get_context(
        conn, str(root), path="src/config_loader.py",
        start_line=2, end_line=4,
    )
    lines = result["source"].splitlines()
    assert len(lines) == 3
    assert lines[0] == "import json" or result["start_line"] == 2


def test_context_defaults_to_120_lines_with_truncation_and_cursor(indexed_repo):
    conn, root = indexed_repo
    big = "\n".join(f"line_{i}" for i in range(1, 401))
    (root / "src" / "big.py").write_text(f"x = 0\n{big}\n")
    result = get_context(conn, str(root), path="src/big.py")
    assert result["truncated"] is True
    assert len(result["source"].splitlines()) == 120
    assert result["next_cursor"]

    page2 = get_context(
        conn, str(root), path="src/big.py", cursor=result["next_cursor"],
    )
    assert page2["start_line"] == 121


def test_path_traversal_outside_root_rejected(indexed_repo):
    conn, root = indexed_repo
    (root.parent / "secret.txt").write_text("secrets")
    with pytest.raises(ContextError):
        get_context(conn, str(root), path="../secret.txt")
    with pytest.raises(ContextError):
        get_context(conn, str(root), path="/etc/passwd")


def test_symlink_escape_rejected(indexed_repo):
    conn, root = indexed_repo
    secret = root.parent / "outside.py"
    secret.write_text("def stolen(): pass\n")
    (root / "src" / "link.py").symlink_to(secret)
    with pytest.raises(ContextError):
        get_context(conn, str(root), path="src/link.py")


def test_entity_context_includes_related_imports(indexed_repo):
    conn, root = indexed_repo
    entity_id = find_function(conn, root, "Config.load")
    result = get_context(
        conn, str(root), entity_type="function", entity_id=entity_id,
    )
    assert result["enclosing"]["qualified_name"] == "Config"
    related_paths = {edge["path"] for edge in result["related"]}
    assert "src/config_loader.py" in related_paths


def test_unknown_entity_raises(indexed_repo):
    conn, root = indexed_repo
    with pytest.raises(ContextError):
        get_context(conn, str(root), entity_type="function", entity_id=99_999)
