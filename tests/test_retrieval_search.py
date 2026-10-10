"""Retrieval search: lexical, symbol-first, semantic, hybrid degradation,
bounds, and cursor invalidation (plan §8.1)."""

import pytest

from indexedcode.retrieval import (
    SearchError,
    StaleCursorError,
    search,
)

from tests.conftest import FakeEmbedder, fake_vector, index_repo


def test_exact_symbol_ranks_first(indexed_repo):
    conn, root = indexed_repo
    result = search(conn, str(root), "parse_config", embedder=FakeEmbedder())
    assert result["results"], "expected at least one result"
    assert result["results"][0]["qualified_name"] == "parse_config"


def test_lexical_search_matches_source_text(indexed_repo):
    conn, root = indexed_repo
    result = search(conn, str(root), "json.loads", mode="lexical")
    names = [r["qualified_name"] for r in result["results"]]
    assert "parse_config" in names


def test_results_carry_paths_lines_and_evidence(indexed_repo):
    conn, root = indexed_repo
    result = search(conn, str(root), "parse_config", mode="lexical")
    top = result["results"][0]
    assert top["path"] == "src/config_loader.py"
    assert top["start_line"] >= 1
    assert "parse_config" in top["evidence"]
    assert top["entity_type"] == "function"
    assert top["entity_id"] >= 1


def test_semantic_mode_without_embedder_fails_clearly(indexed_repo):
    conn, root = indexed_repo
    with pytest.raises(SearchError, match="semantic"):
        search(conn, str(root), "parse json", mode="semantic")


def test_hybrid_without_embedder_degrades_explicitly(indexed_repo):
    conn, root = indexed_repo
    result = search(conn, str(root), "parse_config", mode="hybrid")
    assert result["degraded"] == "semantic_unavailable"
    assert result["results"]


def test_semantic_ranks_identical_source_first(indexed_repo):
    conn, root = indexed_repo
    source = (
        "def format_report(items):\n"
        '    return "\\n".join(str(item) for item in items)\n'
    )
    result = search(
        conn, str(root), source, mode="semantic",
        embedder=lambda texts: [fake_vector(t) for t in texts],
    )
    assert result["degraded"] is None
    assert result["results"][0]["qualified_name"] == "format_report"


def test_limit_bounds_result_count(indexed_repo):
    conn, root = indexed_repo
    result = search(conn, str(root), "e", mode="lexical", limit=2)
    assert len(result["results"]) <= 2
    assert result["truncated"] is True
    assert result["next_cursor"]


def test_cursor_pages_are_disjoint_and_continue(indexed_repo):
    conn, root = indexed_repo
    first = search(conn, str(root), "e", mode="lexical", limit=2)
    second = search(
        conn, str(root), "e", mode="lexical", limit=2,
        cursor=first["next_cursor"],
    )
    ids_first = {(r["entity_type"], r["entity_id"]) for r in first["results"]}
    ids_second = {(r["entity_type"], r["entity_id"]) for r in second["results"]}
    assert ids_first and ids_second
    assert not (ids_first & ids_second)


def test_cursor_invalidated_by_index_change(indexed_repo):
    conn, root = indexed_repo
    first = search(conn, str(root), "e", mode="lexical", limit=1)
    # a new entity_index row must invalidate outstanding cursors
    conn.execute(
        "UPDATE entity_index SET updated_at_ms = updated_at_ms + 1"
        " WHERE id = (SELECT MIN(id) FROM entity_index)"
    )
    conn.commit()
    with pytest.raises(StaleCursorError):
        search(
            conn, str(root), "e", mode="lexical", limit=2,
            cursor=first["next_cursor"],
        )


def test_limit_is_clamped_to_maximum(indexed_repo):
    conn, root = indexed_repo
    result = search(conn, str(root), "e", mode="lexical", limit=10_000)
    assert len(result["results"]) <= 50


def test_garbage_cursor_rejected(indexed_repo):
    conn, root = indexed_repo
    with pytest.raises(SearchError):
        search(conn, str(root), "e", mode="lexical", cursor="not-a-cursor")
