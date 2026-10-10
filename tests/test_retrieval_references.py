"""Relationship search (plan §8.1): bounded, source-grounded, with
resolution state and snippets."""

from indexedcode.retrieval import get_references


def test_references_find_calls_of_symbol(indexed_repo):
    conn, root = indexed_repo
    result = get_references(conn, str(root), "parse_config")
    assert result["references"]
    edges = result["references"]
    assert all(
        edge["relation"] in ("calls", "imports", "references", "inherits")
        for edge in edges
    )
    test_callers = [e for e in edges
                    if e["path"] == "tests/test_config_loader.py"]
    assert test_callers, "expected a reference from the test file"
    edge = test_callers[0]
    assert edge["resolution"] == "syntactic_candidate"
    assert "parse_config" in edge["snippet"]
    assert edge["line_number"] >= 1


def test_references_are_bounded_to_ten(indexed_repo):
    conn, root = indexed_repo
    result = get_references(conn, str(root), "parse_config", limit=100)
    assert len(result["references"]) <= 10


def test_references_include_caller_identity(indexed_repo):
    conn, root = indexed_repo
    result = get_references(conn, str(root), "parse_config")
    callers = {edge.get("source_qualified_name") for edge in result["references"]}
    assert "test_parse_config" in callers or "Config.load" in callers


def test_references_for_unknown_symbol_return_empty(indexed_repo):
    conn, root = indexed_repo
    result = get_references(conn, str(root), "no_such_symbol_anywhere")
    assert result["references"] == []
